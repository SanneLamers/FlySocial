import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import csv
from tqdm import tqdm  # Import the tqdm library

# Create a root Tk window and hide it
root = tk.Tk()
root.withdraw()

# Open a dialog box for the user to select an NPZ file
npz_file_path = filedialog.askopenfilename(filetypes=[("NPZ Files", "*.npz")])

# Load the NPZ file
data = np.load(npz_file_path)

# Get the directory path and the base name of the NPZ file
npz_dir_path = os.path.dirname(npz_file_path)
npz_base_name = os.path.basename(npz_file_path)

# Create a CSV file name based on the NPZ file name
csv_filename = os.path.join(npz_dir_path, f"{npz_base_name}_output.csv")

# Find the maximum number of elements among arrays
max_num_elements = max(len(data[array_name]) for array_name in data.files)

# Open the CSV file for writing
with open(csv_filename, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header row with array names
    header_row = ["Row"] + data.files
    csv_writer.writerow(header_row)
    
    # Use tqdm to display a progress bar
    for i in tqdm(range(max_num_elements), desc="Converting"):
        row_data = [i + 1]  # Start with row number
        
        # Collect data from each array for the current row
        for array_name in data.files:
            array_data = data[array_name]
            if i < len(array_data):
                row_data.append(array_data[i])
            else:
                row_data.append("")  # Fill with empty value if index is out of bounds
        
        # Write the row to CSV
        csv_writer.writerow(row_data)

print(f"Output saved to {csv_filename}")
