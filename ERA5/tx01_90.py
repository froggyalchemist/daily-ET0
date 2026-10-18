"""
tx01_90.py
Calculate the 90th percentile of daily ETos (tx01) for each day of the year based on a 20-year climatology (1981-2000). 
This threshold will be used to identify extreme ETos events in subsequent analyses.
Author: Terence Yeh
Date: 2026-03-31
"""
# %% imported packages
import numpy as np
import xarray as xr
import pandas as pd
# %%
DIRT = r'/work7/IPCS_extreme/share/daily_ETos/ETos_term/'
# latlon = xr.open_dataarray(f'/work7/IPCS_extreme/share/daily_ETos/C_term/C_198101.nc')
data_list = []


for yy in range(1981, 2021):
    for mm in range(1, 13):
        file_path = f'ETos_{yy}{mm:02d}.npy'
        
        # Load the raw numpy array and append it to our list
        arr = np.load(DIRT+file_path)
        data_list.append(arr)
combined_data = np.concatenate(data_list, axis=0)
time_index = pd.date_range(start='1981-01-01', end='2020-12-31', freq='D')

# 4. Construct the xarray DataArrayxx
da_etos = xr.DataArray(
    data=combined_data,
    dims=['time', 'lgrid'],  # Adjust these names if your dimensions are ordered differently
    coords={'time': time_index},
    name='ETos'
)

# Optional: Add some metadata back in
da_etos.attrs['long_name'] = 'Reference Evapotranspiration'
da_etos.attrs['units'] = 'mm/day'
# %%
# 1. Apply your existing mask
mask = ~((da_etos.time.dt.month == 2) & (da_etos.time.dt.day == 29))
ds_no_leap = da_etos.where(mask, drop=True)

# 2. Manually fix the dayofyear coordinate
# Create a custom dayofyear that ignores the leap shift
doy = ds_no_leap.time.dt.dayofyear
is_leap_year = ds_no_leap.time.dt.is_leap_year
after_feb = ds_no_leap.time.dt.month > 2

# Subtract 1 from dayofyear for leap years if the date is after February
custom_doy = doy.where(~(is_leap_year & after_feb), doy - 1)

# 3. Assign this as a coordinate and use it to group
ds_no_leap = ds_no_leap.assign_coords(dayofyear=custom_doy)

tx01_climo = ds_no_leap.sel(time=slice('1981', '2000'))

tx01_90 = (tx01_climo.pad(time=(7,7), mode='wrap')  # Wrap the time series with a 7-day buffer at both ends (Cyclic Padding)
           .rolling(time=15, center=True) 
           .construct('window_dim')
           .isel(time=slice(7, -7))                 # Ensure the 'time' dimension returns to its original length
           .groupby('dayofyear')                    # Group by our custom coord, not time.dayofyear
           .quantile(0.9, dim=['time', 'window_dim']))

tx01_90.to_netcdf("tx01_90_c.nc")
# %% 4m3.5
# da = da_etos
# mask = ~((da.time.dt.month == 2) & (da.time.dt.day == 29))

# # Apply the mask and drop the Feb 29 entries
# ds_no_leap = da.where(mask, drop=True)

# tx01_climo = da.sel(time=slice('1981','2000'))
# tx01_90 = tx01_climo.rolling(time=15,center=True).construct('window_dim').groupby('time.dayofyear').quantile(0.9, dim=['time', 'window_dim'])
# tx01_90.to_netcdf("tx01_90_c.nc")



"""


"""