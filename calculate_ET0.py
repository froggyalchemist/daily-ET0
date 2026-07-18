import numpy as np
import xarray as xr
import cmip6_archive as ca
from rich import print
from rich.console import Console
from datetime import datetime, timezone
from dask.distributed import LocalCluster, progress

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
        raise ValueError(f"Temperature must be in m s-1 but it is in {sfcWind.attrs.get("units")}")
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
        "history":          f"{timestamp}: Created from data at National Taiwan University using ET0-test.ipynb",
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
    ET0_rad = (0.408 * delta * Rn) / denominator              # Assumes G ≈ 0
    ET0_adv = (gamma * 900 * U2 * VPD / (tas_c + 273)) / denominator

    # Calculate potential evapotranspiration as the sum of both terms
    ET0     = ET0_rad + ET0_adv

    # Assign variable-level attributes
    ET0 = ET0.assign_attrs(units="mm day-1", long_name="FAO-56 Penman-Monteith reference evapotranspiration")
    ET0_rad = ET0_rad.assign_attrs(units="mm day-1", long_name="Radiative component of ET0")
    ET0_adv = ET0_adv.assign_attrs(units="mm day-1", long_name="Advective component of ET0")
    VPD = VPD.assign_attrs(units="kPa",     long_name="Vapor pressure deficit")

    # Combine the 4 variables in a single Dataset
    ds = xr.Dataset({
            "ET0":     ET0,
            "ET0_rad": ET0_rad,
            "ET0_adv": ET0_adv,
            "VPD":     VPD,
        })

    # Assign Dataset-level attributes: history, author, parent info, etc.
    ds.attrs = assign_ds_attrs(parent_ds_attrs=parent_ds_attrs)

    return ds

OUTPUTS = ["ET0", "ET0_rad", "ET0_adv", "VPD"]

def process_combination(archive, gcm, exp):
    
    # Safety check
    if gcm not in [model.name for model in ca.GCM_REGISTRY]:
        raise ValueError(f"GCM '{gcm}' is not in CGM_REGISTRY. Modify cmip6_archive.py if necessary.")
    if exp not in ca.EXPERIMENTS:
        raise ValueError(f"Experiment '{exp}' is not in EXPERIMENTS. Modify cmip6_archive.py if necessary.")
    
    # Read in data as Xarray DataArrays
    tas     = archive.get_variable_dataset(gcm, exp, "tas")["tas"]
    ps      = archive.get_variable_dataset(gcm, exp, "ps")["ps"]
    hfls    = archive.get_variable_dataset(gcm, exp, "hfls")["hfls"]
    hfss    = archive.get_variable_dataset(gcm, exp, "hfss")["hfss"]
    sfcWind = archive.get_variable_dataset(gcm, exp, "sfcWind")["sfcWind"]
    hurs    = archive.get_variable_dataset(gcm, exp, "hurs")["hurs"]
    tas_ds  = archive.get_variable_dataset(gcm, exp, "tas")             # Used to copy dataset-level attributes from parent GCM

    # Compute ET0, ET0_rad, ET0_adv, and VPD
    ds = penman_monteith(tas, ps, hfls, hfss, sfcWind, hurs, parent_ds_attrs=tas_ds.attrs).persist()

    # 4 output files in total: ET0, ET0_rad, ET0_adv, and VPD
    paths = []
    for output in OUTPUTS:
        path = f"/work/home/H.mvelasco/SSPs/daily-ET0/test-result/{output}_{gcm}_{exp}.nc"
        ds[[output]].to_netcdf(path) # Save as xr.Dataset
        paths.append(path)

    return paths

if __name__ == "__main__":

    # Dask Cluster to parallelize computations using processes
    cluster = LocalCluster()
    client = cluster.get_client()

    print(f"Started Dask cluster dashboard at {client.dashboard_link}") # Dashboard to monitor computation 
    
    # Create local archive
    archive = ca.CMIP6LocalArchive(root="/work10/archive/CMIP6/CMIP-SSPs/")

    # Test with a single GCM x experiment combo
    gcms = ["MIROC6"]
    exps = ["ssp126", "ssp245"]

    # Compute in parallel
    console = Console()
    with console.status("[bold magenta] Processing...\n", spinner='aesthetic') as status:

        futures = []
        for gcm in gcms:
            for exp in exps:
                future = client.submit(process_combination, archive, gcm, exp)
                futures.append(future)

        results = client.gather(futures)

    #TODO: fix tomorrow --> try with 3 experiments, can't see Dask dashboard for some reason
    #TODO: handle dates (prevent opening unused time data) and set date range at the end of the file
    #TODO: make output path configurable
    #TODO: handle variants

    print("🎉 Finished!")
