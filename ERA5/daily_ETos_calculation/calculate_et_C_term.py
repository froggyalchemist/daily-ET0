import xarray as xr
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.constants import convert_temperature
from numpy import exp, sqrt, log

# Paths
base_path = '/work5/ERA5_daily_landgrid_only/global/'
output_dir = 'C_term'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_month(year, month):
    file_name = f'compressed_sfc_fc_daily_{year}{month:02d}.nc4'
    file_path = os.path.join(base_path, file_name)
    output_file = os.path.join(output_dir, f'C_{year}{month:02d}.npy')
    
    if os.path.exists(output_file):
        print(f'Skipping {output_file}, already exists.')
        return True

    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return False
        
    try:
        ds = xr.open_dataset(file_path)
        
        # T: air temperature at 2-meter (°C)
        T = convert_temperature(ds['2t'], old_scale='Kelvin', new_scale='Celsius')
        
        # Delta: Slope of the saturation vapor pressure curve at a given air temperature (kPa/°C)
        Delta = ( 2503 * exp((17.27*T)/(T+237.3)) ) / ( (T+237.3)**2 )
        
        # P: surface atmospheric pressure (kPa)
        P = ds['sp'] / 1000 # from Pa to kPa
        
        # gamma: Constant multiplied by surface pressure P
        gamma = 0.000665 * P
        
        # u_2: Wind speed at 2-meter (m/s)
        U_10 = sqrt(ds['10u']**2 + ds['10v']**2) # Magnitude of wind speed at 10 m 
        U_2 = U_10 * (4.87 / (log(67.8*10 - 5.42)))
        
        # KPa / K
        C = Delta + gamma*( 1 + 0.34*U_2 )
        
        # Convert to float32 to match previous data and save space
        data = C.values.astype('float32')
        
        # Save to .npy
        np.save(output_file, data)
        ds.close()
        print(f'Successfully processed {output_file}')
        return True
        
    except Exception as e:
        print(f'Error processing {file_name}: {e}')
        return False

def main():
    # Process 43 years: 1981 to 2023
    years = range(1981, 2024)
    months = range(1, 13)
    
    tasks = []
    for year in years:
        for month in months:
            tasks.append((year, month))
            
    print(f'Starting C-term calculation for {len(tasks)} months...')
    
    # Using parallel processing
    max_workers = 12 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_month, year, month): (year, month) for year, month in tasks}
        
        for future in as_completed(futures):
            arg = futures[future]
            try:
                success = future.result()
            except Exception as e:
                print(f'Task {arg} generated an exception: {e}')

if __name__ == '__main__':
    main()
