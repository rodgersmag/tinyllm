import itertools
import pandas as pd
import numpy as np
import time
import os

# Set output path to save locally (modify as needed)
output_path = './euromillions_combinations.csv'

# Function to generate combinations in batches and save to CSV
def generate_euromillions_combinations(batch_size=100000):
    """
    Generate all EuroMillions combinations (N1 to N5, E1, E2) and save to CSV.
    N1 to N5: Unique, sorted integers from 1 to 50.
    E1, E2: Integers from 1 to 12, E1 <= E2.
    Uses batch processing to manage memory.
    """
    start_time = time.time()
    
    # Define main numbers (1 to 50, choose 5) and star numbers (1 to 12, choose 2)
    main_numbers = range(1, 51)
    star_numbers = range(1, 13)
    
    # Generate all star combinations (E1 <= E2)
    star_combos = list(itertools.combinations(star_numbers, 2)) + [(i, i) for i in star_numbers]
    
    # Initialize CSV file with header
    columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2', 'Sum']
    if not os.path.exists(output_path):
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    
    # Generate main number combinations in batches
    main_combos = itertools.combinations(main_numbers, 5)
    batch = []
    batch_count = 0
    
    for main in main_combos:
        main_sum = sum(main)
        for star in star_combos:
            total_sum = main_sum + sum(star)
            batch.append(main + star + (total_sum,))
        
        # Save batch to CSV when it reaches batch_size
        if len(batch) >= batch_size:
            df_batch = pd.DataFrame(batch, columns=columns)
            df_batch.to_csv(output_path, mode='a', header=False, index=False)
            batch = []  # Clear batch
            batch_count += 1
            print(f"Saved batch {batch_count} ({batch_count * batch_size} combinations)")
    
    # Save any remaining combinations
    if batch:
        df_batch = pd.DataFrame(batch, columns=columns)
        df_batch.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Saved final batch ({len(batch)} combinations)")
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")

# Run the generation
generate_euromillions_combinations(batch_size=1000000)

# Verify the output (optional, read a sample)
sample_df = pd.read_csv(output_path, nrows=10)
print("Sample of generated combinations:")
print(sample_df)