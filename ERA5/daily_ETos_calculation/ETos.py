# %%
import xarray as xr
import numpy as np


# %% 
yy = 1996; mm = 5
for mm in range(1,13):
    for yy in range(2021,2024):
        print(yy,mm)
        ## A
        A = np.load(f'/work7/IPCS_extreme/share/daily_ETos/A_term/A_{yy}{mm:02d}.npy')
        A=A.T
        ## C
        C = xr.open_dataarray(f'/work7/IPCS_extreme/share/daily_ETos/C_term/C_{yy}{mm:02d}.nc')
        C = C.values
        ## B
        B = np.load(f'/work7/IPCS_extreme/share/daily_ETos/B_term/B_{yy}{mm:02d}.npy')
        B = B.T
        ETos = A/C + B/C
        np.save(f'/work7/IPCS_extreme/share/daily_ETos/ETos_term/ETos_{yy}{mm:02d}.npy', ETos)
# %%
