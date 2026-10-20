import xarray as xr
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
base_path = '/work5/ERA5_daily_landgrid_only/global/'
output_dir = 'B_term'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_month(year, month):
    file_name = f'compressed_sfc_fc_daily_{year}{month:02d}.nc4'
    file_path = os.path.join(base_path, file_name)
    output_file = os.path.join(output_dir, f'B_{year}{month:02d}.npy')
    
    if os.path.exists(output_file):
        print(f'Skipping {output_file}, already exists.')
        return True

    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return False
        
    try:
        ds = xr.open_dataset(file_path)
        
        # Variables
        temp_k = ds['2t']
        sp_pa = ds['sp']
        dew_k = ds['2d']
        u10 = ds['10u']
        v10 = ds['10v']
        
        # 1. Convert Units
        temp_c = temp_k - 273.15
        dew_c = dew_k - 273.15
        sp_kpa = sp_pa / 1000.0
        
        # 2. Derive Parameters
        # gamma (r) = 0.000665 * P (P in kPa)
        gamma = 0.000665 * sp_kpa
        
        # es = 0.6108 * exp((17.27 * T) / (T + 237.3))
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        
        # ea = 0.6108 * exp((17.27 * T_dew) / (T_dew + 237.3))
        ea = 0.6108 * np.exp((17.27 * dew_c) / (dew_c + 237.3))
        
        # u2 = sqrt(u10^2 + v10^2) * 0.748
        u10_mag = np.sqrt(u10**2 + v10**2)
        u2 = u10_mag * 0.748
        
        # 3. Calculate Term B: gamma * (900 / (T + 273)) * u2 * (es - ea)
        # Note: T+273 is the same as temp_k
        term_b = gamma * (900.0 / temp_k) * u2 * (es - ea)
        
        # Convert to float32 to match previous data and save space
        data = term_b.values.astype('float32')
        
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
            
    print(f'Starting B-term calculation for {len(tasks)} months...')
    
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
