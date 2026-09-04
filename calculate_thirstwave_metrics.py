import numpy as np
import xarray as xr
import warnings
import cmip6_archive as ca
from glob import glob
from pathlib import Path

# Helper functions
def get_growing_month_mask(lat, lon, month):
    """
    Returns a mask that marks whether it's growing season in each grid cell at the given month.
    Ouputs is a boolean array with shape shape len(lat), len(lon).

    *Note: Growing season is Apr-Oct in Northern Hemisphere, Oct-Apr in Southern Hemisphere*
    """
    out = np.zeros(shape=(len(lat), len(lon)), dtype=bool)

    nh = lat >= 0
    sh = lat < 0

    # GS in NH is from April to October
    out[nh] = (4 <= month <= 10)
    # GS in SH is October to December and January to April
    out[sh] = (month >= 10) | (month <= 4)

    return out # array with shape len(lat), len(lon).


def compute_rad_frac(model: str, exp: str) -> xr.Dataset:
    # Load radiative and advective term data
    rad = ca.ET0_RAD_ARCHIVE.open(model, exp, chunks={'time': 365})
    adv = ca.ET0_ADV_ARCHIVE.open(model, exp, chunks={'time': 365})

    # Drop Feb. 29th
    rad = rad.convert_calendar('noleap', align_on='year')["ET0rad"]
    adv = adv.convert_calendar('noleap', align_on='year')["ET0adv"]

    # Daily climatologies from 1981–2000 historical baseline
    rad_hist = ca.ET0_RAD_ARCHIVE.open(model, exp='historical', chunks={'time': 365})
    adv_hist = ca.ET0_ADV_ARCHIVE.open(model, exp='historical', chunks={'time': 365})

    rad_hist = rad_hist.convert_calendar('noleap', align_on='year')["ET0rad"]
    adv_hist = adv_hist.convert_calendar('noleap', align_on='year')["ET0adv"]

    rad_cli = rad_hist.sel(time=slice('1981', '2000')).groupby('time.dayofyear').mean()
    adv_cli = adv_hist.sel(time=slice('1981', '2000')).groupby('time.dayofyear').mean()

    # Compute rad_frac (fractional contribution of radiative term)
    rad_anom = rad.groupby(rad.time.dt.dayofyear) - rad_cli
    adv_anom = adv.groupby(adv.time.dt.dayofyear) - adv_cli
    denom = (rad_anom + adv_anom)
    rad_frac = rad_anom / denom

    # If denominator is negative or rad_frac is infinite, fill grid cell with NaN
    rad_frac = xr.where(cond=((denom <= 0) | (~np.isfinite(rad_frac))),
                        x=np.nan, y=rad_frac)

    return rad_frac.to_dataset(name="rad_frac")

def calculate_event_based_metrics(data_ds: xr.Dataset, mask_ds: xr.Dataset, p90_ds: xr.Dataset, rad_frac_ds: xr.Dataset) -> xr.Dataset:
    mask_da = mask_ds["ET0"]
    data_da = data_ds["ET0"]
    rad_frac_da = rad_frac_ds["rad_frac"]
    p90_da = p90_ds["ET0"]

    # Sanity check: all three must sit on the same grid
    if not all(np.equal(data_da.lat.values, mask_da.lat.values)):
        raise ValueError("mask and data have different latitude values")
    if not all(np.equal(data_da.lon.values, mask_da.lon.values)):
        raise ValueError("mask and data have different longitude values")
    if not all(np.equal(p90_da.lat.values, mask_da.lat.values)):
        raise ValueError("p90 climatology has different latitude values than mask/data")
    if not all(np.equal(p90_da.lon.values, mask_da.lon.values)):
        raise ValueError("p90 climatology has different longitude values than mask/data")

    # The detection algorithm performs bitwise AND and OR operations, which may fail if mask values are integers
    if mask_da.dtype != bool:
        raise ValueError(f"The valid event days mask should contain boolean values but type is {mask_da.dtype}")

    p90_dim0 = p90_da.dims[0]
    if p90_da.sizes[p90_dim0] != 365:
        warnings.warn(f"Expected a 365-day p90 climatology, got {p90_da.sizes[p90_dim0]} days")

    # Remove Feb 29 if needed 
    mask_da = mask_da.convert_calendar('noleap', align_on='year') 
    data_da = data_da.convert_calendar('noleap', align_on='year')

    # Arrays with lat, lon values
    lat, lon = mask_da["lat"].values, mask_da["lon"].values
    n_time, n_lat, n_lon = mask_da.shape

    # Precompute growing season mask by month
    grow_mask_by_month = {m: get_growing_month_mask(lat, lon, m) for m in range(1, 13)}

    years = np.unique(mask_da.time.dt.year.values)
    ny = len(years)

    # These arrays start out empty and end up containing the event-based thirstwave metrics
    sum_event_intensity = np.zeros((ny, n_lat, n_lon), dtype=np.float32)
    event_count = np.zeros((ny, n_lat, n_lon), dtype=np.float32)
    
    sum_event_rad_frac = np.zeros((ny, n_lat, n_lon), dtype=np.float32)
    count_event_rad_frac = np.zeros((ny, n_lat, n_lon), dtype=np.float32)

    true_days = np.zeros((ny, n_lat, n_lon), dtype=np.float32)
    gs_days = np.zeros((ny, n_lat, n_lon), dtype=np.float32)

    for yi, y in enumerate(years):

        yesterday = np.zeros((n_lat, n_lon), dtype=bool) 

        current_event_sum = np.zeros((n_lat, n_lon), dtype=np.float32)
        current_event_count = np.zeros((n_lat, n_lon), dtype=np.float32)

        current_rad_sum = np.zeros((n_lat, n_lon), dtype=np.float32)
        current_rad_count = np.zeros((n_lat, n_lon), dtype=np.float32)

        for m in range(1, 13):

            # Select ET0 data, event days, and 90th percentiles in this month (30 or 31 days)
            data = data_da.sel(time=f"{y}{m:02d}")
            valid_days_month = mask_da.sel(time=f"{y}{m:02d}")
            p90 = p90_da.sel(dayofyear=data.time.dt.dayofyear)
            
            nday = data.shape[0]
            anomaly = data - p90

            # Select fractional contribution of rad term data for this month
            rad_frac = rad_frac_da.sel(time=f"{y}{m:02d}")
            
            # Days this month that are inside a thirstwave event during the growing season 
            valid_days_in_gs = valid_days_month & grow_mask_by_month[m]

            # Sums number of thirswave days in growing season
            true_days[yi, :, :] += valid_days_in_gs.values.sum(axis=0)

            # Sums number of days in growing season 
            # At each grid cell adds 30 or 31 if it is growing season, 0 otherwise
            gs_days[yi, :, :] += (nday * grow_mask_by_month[m].astype(np.float32))

            for di in range(nday):

                today = valid_days_in_gs.isel(time=di).values # is each cell in a thirstwave (and in GS) today?
                anom_today = anomaly.isel(time=di).values # how far above the threshold each cell was today
                
                rad_frac_today = rad_frac.isel(time=di).values # fractional contribution of radiative term in each grid cell today
                finite_today = (today & np.isfinite(rad_frac_today))

                start = today & (~yesterday) # today is TW day and yesterday it wasn't --> Event start
                end = (~today) & yesterday # today is NOT TW day but yesterday it was --> Event end

                valid_end = end & (current_event_count > 0) # positions where events ended

                # Event ended --> add up value of intensity and rad_frac to counters
                sum_event_intensity[yi, valid_end] += (
                    current_event_sum[valid_end]
                    / current_event_count[valid_end]
                )
                event_count[yi, valid_end] += 1

                valid_end_rad = end & (current_rad_count > 0) # positions where events ended

                sum_event_rad_frac[yi, valid_end_rad] += (
                        current_rad_sum[valid_end_rad]
                        / current_rad_count[valid_end_rad]
                )
                count_event_rad_frac[yi, valid_end_rad] += 1

                # Reset current event counters
                current_event_sum[valid_end] = 0
                current_event_count[valid_end] = 0                
                current_rad_sum[valid_end_rad] = 0
                current_rad_count[valid_end_rad] = 0

                # Initialize new events
                current_event_sum[start] = 0
                current_event_count[start] = 0
                current_rad_sum[start] = 0
                current_rad_count[start] = 0

                # Accumulate ongoing events
                current_event_sum[today] += anom_today[today]
                current_event_count[today] += 1
                current_rad_sum[finite_today] += rad_frac_today[finite_today]
                current_rad_count[finite_today] += 1

                yesterday = today

        # finalize events continuing to last day of year
        valid_end = yesterday & (current_event_count > 0)
        sum_event_intensity[yi, valid_end] += (
            current_event_sum[valid_end]
            / current_event_count[valid_end]
        )
        event_count[yi, valid_end] += 1

        valid_end_rad = yesterday & (current_rad_count > 0)
        sum_event_rad_frac[yi, valid_end_rad] += (
                current_rad_sum[valid_end_rad]
                / current_rad_count[valid_end_rad]
        )
        count_event_rad_frac[yi, valid_end_rad] += 1

    # Final event-based metrics
    intensity = sum_event_intensity / event_count
    intensity[event_count == 0] = np.nan

    mean_duration = true_days / event_count
    mean_duration[event_count == 0] = np.nan

    event_freq_100d = event_count / gs_days * 100.0
    event_freq_100d[gs_days == 0] = np.nan

    day_fraction = true_days / gs_days
    day_fraction[gs_days == 0] = np.nan

    rad_frac_tw = np.full((ny, n_lat, n_lon), np.nan, dtype=np.float32)
    valid = count_event_rad_frac > 0
    rad_frac_tw[valid] = sum_event_rad_frac[valid] / count_event_rad_frac[valid]

    # Build output dataset
    coords = ("year", "lat", "lon")
    ds_out = xr.Dataset({
        "intensity": (coords, intensity),
        "event_count": (coords, event_count),
        "mean_duration": (coords, mean_duration),
        "true_days": (coords, true_days),
        "gs_days": (coords, gs_days),
        "event_freq_100d": (coords, event_freq_100d),
        "day_fraction": (coords, day_fraction),
        "rad_frac_tw": (coords, rad_frac_tw),
        },
        coords={"year": years, "lat": lat, "lon": lon}
    )

    # Add metadata
    ds_out["intensity"].attrs["long_name"] = "mean event-based ET0 anomaly during thirstwaves in growing season"
    ds_out["intensity"].attrs["units"] = "mm day-1"

    ds_out["event_count"].attrs["long_name"] = "number of thirstwaves in growing season"
    ds_out["event_count"].attrs["units"] = "events year-1"

    ds_out["mean_duration"].attrs["long_name"] = "mean duration of thirstwave events in growing season"
    ds_out["mean_duration"].attrs["units"] = "days event-1"

    ds_out["true_days"].attrs["long_name"] = "total number of thirstwave days during the growing season"
    ds_out["true_days"].attrs["units"] = "days year-1"

    ds_out["gs_days"].attrs["long_name"] = "length of this year's growing season"
    ds_out["gs_days"].attrs["units"] = "days year-1"

    ds_out["event_freq_100d"].attrs["long_name"] = "thirstwave events per 100 growing-season days"
    ds_out["event_freq_100d"].attrs["units"] = "events per 100 growing-season days"

    ds_out["day_fraction"].attrs["long_name"] = "proportion of growing season days that are thirstwave days"
    ds_out["day_fraction"].attrs["units"] = "%"

    ds_out["rad_frac_tw"].attrs["long_name"] = "mean event-based fractional contribution of radiative term during thirstwaves in growing season"
    ds_out["rad_frac_tw"].attrs["units"] = "1"

    return ds_out

def process_combination(model: str, exp: str):
    try:
        outdir = Path(f"/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/rad_frac/{model}/")
        path = outdir / f"{model}_{exp}_daily_rad_frac_*.nc"
        matches = glob(str(path))

        if len(matches) > 1:
            raise FileExistsError(f"There is more than one rad_frac file for {model} / {exp} at {outdir}")
        if len(matches) == 1:
            # If rad_frac is already computed, load from file
            print(f"[{model} / {exp}] Loading rad_frac from file...")
            outpath = matches[0]

        if not matches:
            # Calculate fractional contribution of radiative term and save result to file
            print(f"[{model} / {exp}] Calculating rad_frac...")
            rad_frac_ds = compute_rad_frac(model, exp)

            outdir.mkdir(exist_ok=True)
            start = rad_frac_ds["time"].dt.strftime("%Y%m%d").values[0]
            end = rad_frac_ds["time"].dt.strftime("%Y%m%d").values[-1]
            outpath = outdir / f"{model}_{exp}_daily_rad_frac_{start}-{end}.nc"
            rad_frac_ds.to_netcdf(outpath)
            print(f"[{model} / {exp}] ✅ SUCCESS: Saved rad_frac to {outpath}")

        # Load valid event days, 90th ET0 percentiles, daily ET0 and fractional
        # contribution of radiative term for this model / experiment
        # We load them as plain NumPy arrays so calculate_event_based_metrics() run faster
        print(f"[{model} / {exp}] Loading daily data...")

        mask_ds = ca.VALID_EVENT_DAYS_ARCHIVE.open(model, exp, chunks=None).load()
        p90_ds = ca.PERCENTILES_ARCHIVE.open(model, exp, chunks=None).load()
        data_ds = ca.ET0_ARCHIVE.open(model, exp, chunks=None).load()
        rad_frac_ds = xr.open_dataset(outpath, chunks=None).load()

        # Calculate event-based metrics and save result to a file
        print(f"[{model} / {exp}] Calculating thirstwave metrics...")
        ds_out = calculate_event_based_metrics(data_ds, mask_ds, p90_ds, rad_frac_ds)

        outdir = Path(f"/work10/archive/CMIP6/CMIP-SSPs/thirstwave_detection/thirstwave_metrics_growing_season/{model}/")
        outdir.mkdir(exist_ok=True)
        outpath = outdir / f"{model}_{exp}_thirstwave_metrics_growing_season.nc"
        ds_out.to_netcdf(outpath)
        print(f"[{model} / {exp}] ✅ SUCCESS: Saved metrics to {outpath}")

        return "SUCCESS"

    except Exception as e:
        print(f"[{model} / {exp}] ❌ ERROR: {e}")
        return "ERROR"

if __name__ == "__main__":

    models = ca.MODELS
    experiments = ca.EXPERIMENTS
    combinations = [(model, exp) for model in models for exp in experiments]

    for model, exp in combinations:
        status = process_combination(model, exp)
        print(f"[{model} / {exp}] STATUS: {status}")

    print("All combinations complete!")





