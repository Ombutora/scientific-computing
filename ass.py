import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# -----------------------------------------------------
# SETUP ENVIRONMENT & DATA EXTRACTION
# -----------------------------------------------------
print("--- 1. Data Extraction & Environment Setup ---")
file_path = r"C:\Users\Chris\Desktop\Youth unemployment, both sexes.xls"
print(f"Loading data from: {file_path}")
df = pd.read_excel(file_path)

# Let's extract the 'Youth unemployed ('000)' column
# We will drop NaNs to avoid math errors during computation
df_clean = df.dropna(subset=["Youth unemployed ('000)"])
unemployed_thousands = df_clean["Youth unemployed ('000)"].tolist()

# The raw dataset has ~2200 rows. To clearly illustrate the benchmarking differences 
# required for scientific computing, we'll magnify the dataset sizes to 1 Million points.
multiplier = 1_000_000 // len(unemployed_thousands)
data_list = unemployed_thousands * multiplier
data_array = np.array(data_list)

print(f"Dataset successfully loaded. Scaled to {len(data_list):,} elements for strict benchmarking.")
print(">> ASSESSMENT:")
print("The raw data is parsed using Pandas. Replicating the values mimics a massive \n"
      "national census dataset, ensuring the gap between naive loops and arrays is evident.\n")


# -----------------------------------------------------
# PROCEDURES IMPLEMENTATION
# -----------------------------------------------------
print("--- 2. Implementing Loop vs Vectorized Logic ---")
# Operation: We want to convert the 'thousands' metric into the absolute number.
# Mathematical Operation: value * 1000

def loop_conversion(data):
    result = []
    # For Loop Iteration over native list
    for value in data:
        result.append(value * 1000.0)
    return result

def vectorized_conversion(data_arr):
    # Direct Scalar mapping on NumPy array
    return data_arr * 1000.0

print("Functions for loop-based and vectorized multiplication cleanly defined.")
print(">> ASSESSMENT:")
print("The Loop approach explicitly parses elements one-by-one using 'for', \n"
      "iteratively scaling the array bounds. The Vectorized approach delegates the math \n"
      "to underlying C operations that automatically map math over the whole vector.\n")


# -----------------------------------------------------
# BENCHMARK PERFORMANCE
# -----------------------------------------------------
print("--- 3. Benchmarking Performance ---")

# Benchmarking the List evaluation
start_loop = time.time()
loop_result = loop_conversion(data_list)
end_loop = time.time()
loop_duration = end_loop - start_loop

# Benchmarking the Array evaluation
start_vect = time.time()
vect_result = vectorized_conversion(data_array)
end_vect = time.time()
vect_duration = end_vect - start_vect

print(f"Execution Time (Loop):       {loop_duration:.5f} seconds")
print(f"Execution Time (Vectorized): {vect_duration:.5f} seconds")
speedup = loop_duration / vect_duration if vect_duration > 0 else float('inf')

print(">> CONCLUSION:")
print(f"The NumPy Vectorized approach was ~{speedup:.2f}x faster on the dataset.")
print(">> ASSESSMENT:")
print("Native Python loops suffer heavily from 'interpreter overhead' since Python has \n"
      "to dynamically check the datatype of every item in the loop. NumPy entirely bypasses \n"
      "this, resulting in dramatically higher execution speeds.\n")


# -----------------------------------------------------
# MEASURE MEMORY USAGE
# -----------------------------------------------------
print("--- 4. Measuring Memory Usage ---")

# Python lists store metadata + object pointers + the objects themselves
list_mem = sys.getsizeof(data_list) + sum(sys.getsizeof(item) for item in data_list)

# NumPy arrays are strictly packed C-data blocks in memory
array_mem = sys.getsizeof(data_array)

print(f"Memory Usage (Python List):  {list_mem:,} bytes")
print(f"Memory Usage (NumPy Array):  {array_mem:,} bytes")
mem_ratio = list_mem / array_mem if array_mem > 0 else float('inf')

print(">> CONCLUSION:")
print(f"Python Lists consumed ~{mem_ratio:.2f}x more RAM footprint.")
print(">> ASSESSMENT:")
print("Lists are unstructured, forcing objects to be scattered in Memory. In memory \n"
      "restricted systems (large Big Data tasks), NumPy provides massive efficiency \n"
      "because array elements are typed sequentially as adjacent bytes without structural bloat.\n")


# -----------------------------------------------------
# VISUALIZE RESULTS
# -----------------------------------------------------
print("--- 5. Visualizing Results ---")
print(">> ASSESSMENT:")
print("Using Matplotlib brings analytical contexts to benchmarking results. (Look for the \n"
      "chart pop-up interface!).")

# Setting up dual bar charts to show final conclusions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Processing Time
axes[0].bar(["Loop (List)", "Vectorized (NumPy)"], [loop_duration, vect_duration], color=['salmon', 'mediumseagreen'])
axes[0].set_title('Processing Speed Comparison')
axes[0].set_ylabel('Execution Time (seconds)')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Subplot 2: Memory Overhead
axes[1].bar(["Loop (List)", "Vectorized (NumPy)"], [list_mem / (1024**2), array_mem / (1024**2)], color=['orange', 'cornflowerblue'])
axes[1].set_title('RAM Footprint Comparison')
axes[1].set_ylabel('Memory Overhead (Megabytes)')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Processing Youth Unemployment Statistics (1M Entries)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n** Execution Complete **")
