import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
base_dir = '.'
output_dir = 'ETos'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_month(year, month):
    a_file = os.path.join(base_dir, 'A_term', f'A_{year}{month:02d}.npy')
    b_file = os.path.join(base_dir, 'B_term', f'B_{year}{month:02d}.npy')
    c_file = os.path.join(base_dir, 'C_term', f'C_{year}{month:02d}.npy')
    output_file = os.path.join(output_dir, f'ETos_{year}{month:02d}.npy')
    
    if os.path.exists(output_file):
        return f'Skipped {year}-{month:02d}'
        
    if not (os.path.exists(a_file) and os.path.exists(b_file) and os.path.exists(c_file)):
        return f'Warning: Missing files for {year}-{month:02d}'
        
    try:
        a = np.load(a_file)
        b = np.load(b_file)
        c = np.load(c_file)
        
        # Calculate ETos = (A + B) / C
        # Division is safe as C (~0.2) is the denominator (Delta + gamma*)
        etos = (a + b) / c
        
        # Save as float32
        np.save(output_file, etos.astype('float32'))
        return f'Successfully calculated ETos for {year}-{month:02d}'
    except Exception as e:
        return f'Error processing {year}-{month:02d}: {e}'

def main():
    # Process 43 years: 1981 to 2023
    years = range(1981, 2024)
    months = range(1, 13)
    tasks = [(y, m) for y in years for m in months]
    
    print(f'Starting ETos calculation for {len(tasks)} months...')
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_month, y, m): (y, m) for y, m in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                print(res)

if __name__ == "__main__":
    main()
