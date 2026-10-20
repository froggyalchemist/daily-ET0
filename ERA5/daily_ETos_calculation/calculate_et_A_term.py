import xarray as xr
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
base_path = '/work5/ERA5_daily_landgrid_only/global/'
output_dir = 'A_term'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_month(year, month):
    file_name = f'compressed_sfc_fc_daily_{year}{month:02d}.nc4'
    file_path = os.path.join(base_path, file_name)
    output_file = os.path.join(output_dir, f'A_{year}{month:02d}.npy')
    
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
        msnswrf = ds['msnswrf']
        msnlwrf = ds['msnlwrf']
        
        # 1. Convert Temp to Celsius
        temp_c = temp_k - 273.15
        
        # 2. Calculate Net Radiation Rn in MJ m-2 day-1
        # conversion_factor = 0.0864 (from W/m^2 to MJ/m^2/day)
        rn_mj = (msnswrf + msnlwrf) * 0.0864
        
        # 3. Calculate Delta (slope of vapor pressure curve)
        delta = (2503 * np.exp((17.27 * temp_c) / (temp_c + 237.3))) / ((temp_c + 237.3)**2)
        
        # 4. Target Value (A term): 0.408 * Delta * Rn
        a_term = 0.408 * delta * rn_mj
        
        # Convert to numpy array (float32 to save space)
        data = a_term.values.astype('float32')
        
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
            
    print(f'Starting A-term calculation for {len(tasks)} months (NPY format)...')
    
    # Using 12 workers for parallel processing
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
