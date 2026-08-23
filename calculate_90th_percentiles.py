import numpy as np
import pandas as pd
import xarray as xr
import cmip6_archive as ca
from pathlib import Path
from datetime import datetime, timezone
from xarray.groupers import UniqueGrouper
from rich import print
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich_tools import df_to_table


def preprocess_data(da: xr.DataArray):
    """
    Prepare a daily ET0 DataArray for percentile calculation.

    Drops Feb 29th so every year has 365 days, tags each timestep with its
    Julian day of year (1-365) as a 'dayofyear' coordinate, subsets the data
    to the reference/baseline period (1981-2000), and rechunks it so the full
    time series is contiguous per chunk while lat/lon are split into smaller
    chunks.
    """
    # Remove all February 29ths
    da_noleap = da.convert_calendar("noleap", 
                                    align_on="year") # Necessary because UKESM1-0-LL has a 360 calendar and needs to be converted to 365 days

    # Add 'dayofyear' coordinate (1 = Jan. 1st, ... , 365 = Dec. 31st)
    doy = da_noleap.time.dt.dayofyear
    da_noleap = da_noleap.assign_coords(dayofyear=doy)

    # Reference period only (1981-2000)
    climo = da_noleap.sel(time=slice("1981", "2000"))
    climo = climo.chunk(chunks={"time": -1, "lat": 8, "lon": 16})

    return climo


def get_90th_percentiles(climo: xr.DataArray):
    """
    **climo** is a DataArray containing the values of ET0 over the baseline period (1981-2000). It should have a 'dayofyear' coordinate [1, 2, 3, ... , 365]

    Computes the 90th percentile of ET0 for each Julian day. The 90th percentile is defined using
    a 15-day moving window centered on the Julian day being evaluated over the baseline period (1981-2000).

    For example, to find the 90th percentile for April 15th we:
    1. Gather ETos data for April 1st - April 30th for 1981 to 2000. We'll have $30 \times 20 = 600$ daily observations.
    2. Find the 90th percentile of these values. Any value of ET0 on April 15th larger than this is considered "extreme".
    """
    climo = climo.load()

    percentiles = (
        climo.pad(
            time=(7, 7), mode="wrap"
        )  # Wrap the time series with a 7-day buffer at both ends (Cyclic Padding)
        .rolling(time=15, center=True)
        .construct("window_dim")
        .isel(
            time=slice(7, -7)
        )  # Ensure the 'time' dimension returns to its original length
        .groupby(
            dayofyear=UniqueGrouper(labels=np.arange(1, 366))
        )  # Group by dayofyear
        .quantile(0.9, dim=["time", "window_dim"])
    )

    return percentiles


def process_model(model: str, experiment: str, input_dir: str, output_dir: str) -> str:
    """
    Load daily ET0 for a single GCM/experiment, compute the 90th-percentile
    threshold for every day of the year over the 1981-2000 baseline period,
    and save the result to a netCDF file. Returns the path of the saved file.
    """
    path = (
        Path(input_dir)
        / model
        / experiment
        / f"{model}_{experiment}_daily_ET0_18500101-20141231.nc"
    )

    # Historical simulation of this model ends on 2014-12-30
    if model == 'UKESM1-0-LL':
        path = "/work10/archive/CMIP6/CMIP-SSPs/outputs/UKESM1-0-LL/historical/UKESM1-0-LL_historical_daily_ET0_18500101-20141230.nc"

    # Compute 90th percentiles
    da = xr.open_dataset(path).ET0
    climo = preprocess_data(da)
    climo = climo.load()
    percentiles = get_90th_percentiles(climo=climo)

    # Save output to file
    out_path = Path(output_dir) / f"{model}_{experiment}_90th_percentiles.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    percentiles.to_netcdf(out_path)

    return str(out_path)


if __name__ == "__main__":

    # The reference/baseline period (1981-2000) is only defined for the historical
    # experiment, so this is always run with experiment="historical".
    experiment = "historical"
    input_dir = "/work10/archive/CMIP6/CMIP-SSPs/outputs"
    output_dir = "/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/90th_percentiles"

    # Compute the 90th percentile thresholds for every GCM in the registry
    #models = [config.name for config in ca.GCM_REGISTRY]
    models = ['UKESM1-0-LL']

    console = Console()
    log_rows = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Calculating 90th percentiles..."),
        BarColumn(),
        MofNCompleteColumn(),  # models processed so far
        TimeElapsedColumn(),  # time passed since the run started
        console=console,
    ) as progress_bar:

        task = progress_bar.add_task("models", total=len(models))

        for model in models:
            try:
                out_path = process_model(
                    model=model,
                    experiment=experiment,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )
                log_rows.append(
                    {
                        "model": model,
                        "experiment": experiment,
                        "status": "✅ success",
                        "error": None,
                        "output_file": out_path,
                    }
                )
            except Exception as e:
                log_rows.append(
                    {
                        "model": model,
                        "experiment": experiment,
                        "status": "☠️ failed",
                        "error": f"{type(e).__name__}: {e}",
                        "output_file": None,
                    }
                )
            progress_bar.advance(task)

    print("🎉 Finished!")

    # Write the run log to a csv, detailing which models succeeded and why the rest failed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(f"./logs/calculate_90th_percentiles_log_{timestamp}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False)

    # Display the run log as a rich table
    table = df_to_table(log_df, show_index=False)
    console.print(table)
    console.print(f"\nRun log written to {log_path}")
