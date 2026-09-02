import xarray as xr
import cmip6_archive as ca
from pathlib import Path

def find_et0_file(model: str, exp: str, var: str, base_dir = Path("/work10/archive/CMIP6/CMIP-SSPs/outputs")) -> Path:
    """Find the ET0 (not ET0adv/ET0rad/VPD) file for a model/experiment.

    Uses a glob instead of a hardcoded date range because some models
    (e.g. UKESM1-0-LL, which runs a 360-day calendar) end their filenames
    in a different date than the rest (...1230 instead of ...1231).
    """
    matches = sorted(base_dir.glob(f"{model}/{exp}/{model}_{exp}_daily_{var}_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No {var} file found in {base_dir} for {model}/{exp}")
    return matches[0]

def find_percentiles_file(model: str, base_dir = Path("/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/90th_percentiles")) -> Path:
    """Find the ET0 90th percentile file for a model."""
    matches = sorted(base_dir.glob(f"{model}_historical_90th_percentiles.nc"))
    if not matches:
        raise FileNotFoundError(f"No percentiles file found in {base_dir} for {model}")
    return matches[0]

def preprocess_ET0(ds: xr.Dataset) -> xr.DataArray:
    """Loads daily ET0 for the specified model and experiment, converts to 365-day calendar and adds a 'dayofyear' coordinate."""   
    
    # Remove Feb. 29th
    ds = ds.convert_calendar('noleap', align_on='year')
    
    # Add 'dayofyear' coordinate (1 = Jan. 1st, ... , 365 = Dec. 31st)
    # Note that UKESM1-0-LL has a 360-day calendar, but 'dayofyear' still ranges between 1 and 365
    ds = ds.assign_coords(dayofyear=ds.time.dt.dayofyear)
    
    return ds.ET0
    

def calculate_valid_event_days(et0: xr.DataArray, p90: xr.Dataset) -> xr.Dataset:
    """Returns a Dataset where (time, lat, lon) = True if that day there was a thirstwave at that (lat, lon), False otherwise"""
    
    # Days where ETos is above the threshold
    above = et0.groupby(et0.dayofyear) > p90
    
    # Identify >=3 consecutive days
    consecutive_3 = above.rolling(time=3, center=False).sum() == 3
    valid_event_days = (
        consecutive_3
        | consecutive_3.shift(time=-1, fill_value=False)
        | consecutive_3.shift(time=-2, fill_value=False)
    )
    return valid_event_days

def calculate_thirstwave_stats(et0: xr.DataArray, p90: xr.Dataset, valid_event_days: xr.Dataset) -> xr.Dataset:
    """Measures thirstwave frequency, intensity and duration from the mask of valid event days
    (one value per grid cell and year) """

    # A day is the start of an event if it's flagged but the day before it wasn't
    event_starts = valid_event_days & ~valid_event_days.shift(time=1, fill_value=False)
    
    # Frequency = number of events each year (year⁻¹)
    frequency = event_starts.astype(int).groupby('time.year').sum('time')
    
    # Intensity = mean ETos anomaly above the 90th-percentile threshold during thirstwave days (mm/day)
    # TODO: change to event-based calculation --> calculate intensity of each event 1st (mean(event_anomalies) / n_event_days), then average along the year
    anomaly = et0.groupby(et0.dayofyear) - p90
    intensity = anomaly.where(valid_event_days).groupby('time.year').mean('time')
    
    # Duration = mean thirstwave event duration per year (days)
    event_days_per_year = valid_event_days.astype(int).groupby('time.year').sum('time')
    duration = event_days_per_year / frequency.where(frequency > 0)
    
    # Save metrics to file
    thirstwave_stats = xr.Dataset({'frequency': frequency.ET0,
                                   'duration': duration.ET0,
                                   'intensity': intensity.ET0
                                   })
    thirstwave_stats['frequency'].attrs = {'long_name': 'Number of thirstwave events per year',
                                           'units': 'count year-1'
                                           }
    thirstwave_stats['duration'].attrs = {'long_name': 'Mean thirstwave event duration per year',
                                          'units': 'days'}
    thirstwave_stats['intensity'].attrs = {'long_name': 'Mean ETos anomaly above 90th percentile during thirstwave days',
                                           'units': 'mm day-1'}
    return thirstwave_stats


if __name__ == "__main__":

    experiments = ca.EXPERIMENTS
    models = [m.name for m in ca.GCM_REGISTRY]

    for model in models:
        for exp in experiments:
            try:
                print(f"[{model} / {exp}] Identifying thirstwaves...")
                
                # Load and preprocess ET0 data
                file = find_et0_file(model, exp, var = 'ET0')
                ds = xr.open_dataset(file)
                et0 = preprocess_ET0(ds)

                # Load ETos 90th percentile for this model
                file_p90 = find_percentiles_file(model)
                p90 = xr.open_dataset(file_p90)
                
                # Calculate valid event days and save to a file
                valid_event_days = calculate_valid_event_days(et0, p90)
                outdir = Path("/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/valid_event_days")
                years = valid_event_days.time.dt.year.values
                filename = outdir / f"{model}_{exp}_valid_event_days_{years[0]}-{years[-1]}.nc"
                print(f"[{model} / {exp}] Saving valid event days to {filename} ...")
                valid_event_days.to_netcdf(filename, engine='netcdf4')

                # Calculate annual thirswave frequency, magnitude and duration, and save to a file
                thirstwave_stats = calculate_thirstwave_stats(et0, p90, valid_event_days)
                outdir_stats = Path("/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/thirstwave_features")
                filename_stats = outdir_stats / f"{model}_{exp}_thirstwave_features_{years[0]}-{years[-1]}.nc"
                print(f"[{model} / {exp}] Saving thirstwave features to {filename_stats} ...")
                thirstwave_stats.to_netcdf(filename_stats, engine='netcdf4')
                
            except Exception as e:
                print(f"[{model} / {exp}] ERROR {e}")

    print("Done!")
