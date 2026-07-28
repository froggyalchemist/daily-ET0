import numpy as np
import pandas as pd
import xarray as xr
import cmip6_archive as ca
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich_tools import df_to_table
from pathlib import Path
from datetime import datetime, timezone
from dask.distributed import LocalCluster, as_completed

def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert temperature DataArray from K to °C if needed."""
    if da.attrs.get("units") == "K":
        da.attrs['units'] = "C"
        return da - 273.15
    else:
        raise AttributeError(f"Temperature must be in Kelvin but units are {da.attrs.get("units")}")

def net_surface_radiation(hfls: xr.DataArray, hfss: xr.DataArray) -> xr.DataArray:
    """Rn = hfls + hfss (W/m²), converted to MJ/m²/day. """
    if hfls.attrs.get("units") == "W m-2" and hfss.attrs.get("units") == "W m-2":
        return (hfls + hfss) * 86400 / 1e6 
    else:
        raise ValueError(f"Inputs must be in W m-2 but units are {hfls.attrs.get("units")}, {hfss.attrs.get("units")}")

def saturation_vapor_pressure(tas: xr.DataArray) -> xr.DataArray:
    """es in kPa from air temperature in °C. tas must be in °C."""
    if tas.attrs.get("units") != "C":
        raise ValueError(f"Temperature must be in ºC but it is in {tas.attrs.get("units")}")
    return 0.6108 * np.exp((17.27 * tas) / (tas + 237.3))

def actual_vapor_pressure(hurs: xr.DataArray, es: xr.DataArray) -> xr.DataArray:
    """ea in kPa from relative humidity (%) and saturation vapor pressure (es).
    Source: Eq. 19 FAO56 Guidelines (https://www.fao.org/4/x0490e/x0490e07.htm)"""
    return (hurs / 100.0) * es

def slope_saturation_vapor_pressure_curve(tas: xr.DataArray) -> xr.DataArray:
    """delta in kPa/°C tas must be in °C."""
    if tas.attrs.get("units") != "C":
        raise ValueError(f"Temperature must be in ºC but it is in {tas.attrs.get("units")}")

    return ( 2503 * np.exp((17.27 * tas) / (tas + 237.3)) ) / ((tas + 237.3) ** 2)

def psychrometric_constant(ps: xr.DataArray) -> xr.DataArray:
    """gamma in kPa/°C from surface pressure in Pa."""
    if ps.attrs.get("units") != "Pa":
        raise ValueError(f"Pressure must be in Pa but it is in {ps.attrs.get("units")}")
    return 0.000665 * (ps / 1000)

def wind_speed_2m(sfcWind: xr.DataArray, height: float = 10.0) -> xr.DataArray:
    """Rescale wind from measurement height to 2 m using FAO-56 log profile."""
    if sfcWind.attrs.get("units") != "m s-1":
        raise ValueError(f"Wind speed must be in m s-1 but it is in {sfcWind.attrs.get("units")}")
    return sfcWind  * 4.87 / np.log(67.8 * height - 5.42)

def assign_ds_attrs(parent_ds_attrs: dict) -> dict:
    """
    Add dataset-level attributes: history, authors, source_id, variant_label, and experiment_id
    #TODO decide which attributes from parent dataset to keep
    """
    timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
    attrs = {
        "creation_date":    timestamp,
        "history":          f"{timestamp}: Created from data at National Taiwan University using calculate_ET0.py",
        "authors":          "Marina Velasco-Barriuso (UPF)",
        "source_id":        parent_ds_attrs["source_id"],
        "variant_label":    parent_ds_attrs["variant_label"],
        "experiment_id":    parent_ds_attrs["experiment_id"],
    }

    return attrs

def penman_monteith(
    tas:      xr.DataArray,
    ps:       xr.DataArray,
    hfls:     xr.DataArray,
    hfss:     xr.DataArray,
    sfcWind:  xr.DataArray,
    hurs:     xr.DataArray,
    parent_ds_attrs: dict,
    height:   float = 10.0,
) -> xr.Dataset:
    """
    FAO-56 Penman-Monteith ET0 (mm/day). 
    Returns a Dataset with ET0, ET0_rad, ET0_adv, and VPD.
    """
    tas_c = kelvin_to_celsius(tas)

    # Calculate necessary variables
    # These functions convert units if needed
    Rn    = net_surface_radiation(hfls, hfss)
    delta = slope_saturation_vapor_pressure_curve(tas_c)
    gamma = psychrometric_constant(ps)
    U2    = wind_speed_2m(sfcWind, height=height)
    es  = saturation_vapor_pressure(tas_c)
    ea    = actual_vapor_pressure(hurs, es)

    # Calculate Vapor Pressure Deficit
    VPD   = es - ea

    # Calculate radiative and advective components separately
    denominator   = delta + gamma * (1 + 0.34 * U2)            # Common to both terms
    ET0rad = (0.408 * delta * Rn) / denominator              # Assumes G ≈ 0
    ET0adv = (gamma * 900 * U2 * VPD / (tas_c + 273)) / denominator

    # Calculate potential evapotranspiration as the sum of both terms
    ET0     = ET0rad + ET0adv

    # Assign variable-level attributes
    ET0.attrs = {"units": "mm day-1",
                 "long_name": "FAO-56 Penman-Monteith reference evapotranspiration"}
    ET0rad.attrs = {"units": "mm day-1",
                     "long_name": "Radiative component of ET0"}
    ET0adv.attrs = {"units": "mm day-1",
                     "long_name": "Advective component of ET0"}
    VPD.attrs = {"units": "kPa",
                 "long_name": "Vapor pressure deficit"}
    
    # Combine the 4 variables in a single Dataset
    ds = xr.Dataset({
            "ET0":     ET0,
            "ET0rad": ET0rad,
            "ET0adv": ET0adv,
            "VPD":     VPD,
        })

    # Assign Dataset-level attributes: history, author, parent info, etc.
    ds.attrs = assign_ds_attrs(parent_ds_attrs=parent_ds_attrs)

    return ds

# Default location for output netCDFs (and the run log)
#DEFAULT_OUTPUT_DIR = "/work/home/H.mvelasco/SSPs/daily-ET0/test-result"
DEFAULT_OUTPUT_DIR = "/work10/archive/CMIP6/CMIP-SSPs/outputs"

def process_combination(archive, gcm, exp, chunks, output_dir: str = DEFAULT_OUTPUT_DIR):
    
    # Safety check
    if gcm not in [model.name for model in ca.GCM_REGISTRY]:
        raise ValueError(f"GCM '{gcm}' is not in CGM_REGISTRY. Modify cmip6_archive.py if necessary.")
    if exp not in ca.EXPERIMENTS:
        raise ValueError(f"Experiment '{exp}' is not in EXPERIMENTS. Modify cmip6_archive.py if necessary.")
    
    # Read in data as Xarray DataArrays
    tas_ds  = archive.get_variable_dataset(gcm, exp, "tas", chunks=chunks)             # Used to copy dataset-level attributes from parent GCM
    tas     = tas_ds["tas"]
    ps      = archive.get_variable_dataset(gcm, exp, "ps", chunks=chunks)["ps"]
    hfls    = archive.get_variable_dataset(gcm, exp, "hfls", chunks=chunks)["hfls"]
    hfss    = archive.get_variable_dataset(gcm, exp, "hfss", chunks=chunks)["hfss"]
    sfcWind = archive.get_variable_dataset(gcm, exp, "sfcWind", chunks=chunks)["sfcWind"]
    hurs    = archive.get_variable_dataset(gcm, exp, "hurs", chunks=chunks)["hurs"]

    # Compute Dataset with ET0, ET0_rad, ET0_adv, and VPD
    ds = penman_monteith(tas, ps, hfls, hfss, sfcWind, hurs, parent_ds_attrs=tas_ds.attrs).persist()

    # Make sure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Start and end days as strings (e.g. '20191231')
    start = ds["time"].dt.strftime("%Y%m%d").values[0]
    end = ds["time"].dt.strftime("%Y%m%d").values[-1]

    # 4 output files in total: ET0, ET0_rad, ET0_adv, and VPD
    paths = []
    for var in ds.data_vars:
        path = f"{output_dir}/{var}_day_{gcm}_{exp}_{start}-{end}.nc"
        ds[[var]].to_netcdf(path) # Save as xr.Dataset
        paths.append(path)

    return paths

if __name__ == "__main__":

    # Dask Cluster to parallelize computations using processes
    cluster = LocalCluster()
    client = cluster.get_client()

    print(f"Started Dask cluster dashboard at {client.dashboard_link}") # Dashboard to monitor computation 
    
    # Create local archive
    archive = ca.CMIP6LocalArchive(root="/work10/archive/CMIP6/CMIP-SSPs/")

    # All models, historical simulation
    gcms = ["IPSL-CM6A-LR"]
    #gcms = [model.name for model in ca.GCM_REGISTRY] # Use all GCMs
    exps = ["historical"]
    combinations = [(gcm, exp) for gcm in gcms for exp in exps]

    # Show a progress bar with total combinations completed
    log_rows = []
    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Processing combinations..."),
        BarColumn(),
        MofNCompleteColumn(),          # combinations processed so far
        TimeElapsedColumn(),           # time passed since the run started
        console=console,
    ) as progress_bar:

        task = progress_bar.add_task("combinations", total=len(combinations))

        # Compute GCM x experiment combinations sequentially
        for (gcm, exp) in combinations:
            try:
                paths = process_combination(archive, gcm, exp, chunks = {'time': 5*365})
                log_rows.append({
                    "gcm": gcm,
                    "experiment": exp,
                    "status": "✅ success",
                    "error": None,
                    "output_files": ", ".join(paths),
                }) 
            except Exception as e:
                log_rows.append({
                    "gcm": gcm,
                    "experiment": exp,
                    "status": "☠️ failed",
                    "error": f"{type(e).__name__}: {e}",
                    "output_files": None,
                })
            progress_bar.advance(task)

    # Close the client
    client.close()

    print("🎉 Finished!")

    # Write the run log to a csv, detailing which combinations succeeded and why the rest failed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(DEFAULT_OUTPUT_DIR) / 'logs' / f"calculate_ET0_log_{timestamp}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False)

    # Display the run log as a rich table
    table = df_to_table(log_df, show_index=False)
    console.print(table)
    console.print(f"\nRun log written to {log_path}")
