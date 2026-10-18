# %% imported packages
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# %% read ETos
DIRT = r'/work7/IPCS_extreme/share/daily_ETos/ETos_term/'
data_list = []

for yy in range(1981, 2024):
    for mm in range(1, 13):
        file_path = f'ETos_{yy}{mm:02d}.npy'
        arr = np.load(DIRT + file_path)
        data_list.append(arr)

combined_data = np.concatenate(data_list, axis=0)
time_index = pd.date_range(start='1981-01-01', end='2023-12-31', freq='D')

da_etos = xr.DataArray(
    data=combined_data,
    dims=['time', 'lgrid'],
    coords={'time': time_index},
    name='ETos'
)

da_etos.attrs['long_name'] = 'Reference Evapotranspiration'
da_etos.attrs['units'] = 'mm/day'

# =========================================================
# Remove Feb 29 first
# =========================================================
da = da_etos.copy()

is_feb29 = (da.time.dt.month == 2) & (da.time.dt.day == 29)
da = da.sel(time=~is_feb29)

# Build a 365-day day-of-year index
doy = da.time.dt.dayofyear.values.copy()
is_leap = da.time.dt.is_leap_year.values
month = da.time.dt.month.values
day = da.time.dt.day.values

after_feb28_in_leap = is_leap & ((month > 2) | ((month == 2) & (day > 28)))
doy[after_feb28_in_leap] = doy[after_feb28_in_leap] - 1

da = da.assign_coords(dayofyear_365=('time', doy))

# =========================================================
# Baseline and threshold
# =========================================================
tx01_climo = da.sel(time=slice('1981', '2000'))
tx01 = da.sel(time=slice('1981', '2023'))

# If your threshold file already has 365 days, use it directly
tx01_90 = xr.open_dataarray("/work7/IPCS_extreme/B11209028/tx01_90_c.nc")

# Make sure threshold dimension name is dayofyear_365
# If original dimension is 'dayofyear', rename it
if 'dayofyear' in tx01_90.dims:
    tx01_90 = tx01_90.rename({'dayofyear': 'dayofyear_365'})

# Optional safety check
if tx01_90.sizes['dayofyear_365'] != 365:
    raise ValueError(f"tx01_90 should have 365 days, got {tx01_90.sizes['dayofyear_365']}")

# =========================================================
# 1. Boolean mask: ETos > 90th percentile
# =========================================================
mask = tx01.groupby('dayofyear_365') > tx01_90

# =========================================================
# 2. Identify >=3 consecutive days
# =========================================================
consecutive_3 = mask.rolling(time=3, center=False).sum() == 3

valid_event_days = (
    consecutive_3
    | consecutive_3.shift(time=-1, fill_value=False)
    | consecutive_3.shift(time=-2, fill_value=False)
)

valid_event_days.to_netcdf("/work7/IPCS_extreme/HH/valid_event_days_1981_2023_new.nc")

# =========================================================
# Frequency
# =========================================================
event_starts = valid_event_days.astype(int).diff(dim='time') == 1
first_day_starts = valid_event_days.isel(time=0)
total_events = event_starts.sum(dim='time') + first_day_starts.astype(int)

num_years = len(np.unique(tx01.time.dt.year))
frequency = total_events / num_years

# =========================================================
# Duration
# =========================================================
total_event_days = valid_event_days.sum(dim='time')
mean_duration = total_event_days / total_events.where(total_events > 0)

# =========================================================
# Magnitude
# =========================================================
daily_anomaly = tx01.groupby('dayofyear_365') - tx01_90
event_magnitudes = daily_anomaly.where(valid_event_days)
mean_daily_magnitude = event_magnitudes.mean(dim='time')

# =========================================================
# Save
# =========================================================
frequency.to_netcdf('ETos_frequency1.nc')
mean_duration.to_netcdf('ETos_duration1.nc')
mean_daily_magnitude.to_netcdf('ETos_magnitude1.nc')
