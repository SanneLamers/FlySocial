import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
import tkinter as tk
from tkinter import filedialog

# Two functions for reading a csv or npz input file respectively
def read_csv_file(file_path):
    """
    Reads the coordinate data and angle and midline_length from a CSV file and returns the X and Y coordinates,
    ANGLE, midline_length, and a Boolean mask indicating missing data points.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        tuple: A tuple containing the X and Y coordinate arrays, ANGLE, midline_length, and the missing data Boolean mask.
    """
    df = pd.read_csv(file_path)

    # Use regular expressions to find the columns
    x_col = df.columns[df.columns.str.match(r'X#wcentroid(\s+\(cm\))?')][0]
    y_col = df.columns[df.columns.str.match(r'Y#wcentroid(\s+\(cm\))?')][0]
    angle_col = df.columns[df.columns.str.match(r'ANGLE')][0]
    midline_length_col = df.columns[df.columns.str.match(r'midline_length')][0]
    frame_col = df.columns[df.columns.str.match(r'frame')][0]

    X = df[x_col].values
    Y = df[y_col].values
    ANGLE = df[angle_col].values
    midline_length = df[midline_length_col].values
    missing = df['missing'].astype(bool).values
    frame_numbers = df[frame_col].values
    
    return X, Y, missing, ANGLE, midline_length, frame_numbers
def read_npz_file(file_path):
    """
    Reads the coordinate data and angle and midline_length from an NPZ file and returns the X and Y coordinates,
    ANGLE, midline_length, and a Boolean mask indicating missing data points.
    
    Args:
        file_path (str): The path to the NPZ file.
        
    Returns:
        tuple: A tuple containing the X and Y coordinate arrays, ANGLE, midline_length, and the missing data Boolean mask.
    """
    with np.load(file_path) as npz:
        keys = npz.files
        X_key = [key for key in keys if re.match(r'X#wcentroid(\s+\(cm\))?$', key)][0]
        Y_key = [key for key in keys if re.match(r'Y#wcentroid(\s+\(cm\))?$', key)][0]
        ANGLE_key = [key for key in keys if 'ANGLE' in key][0]
        midline_length_key = [key for key in keys if 'midline_length' in key][0]
        missing_key = [key for key in keys if 'missing' in key][0]
        frame_key = [key for key in keys if 'frame' in key][0]

        X = npz[X_key]
        Y = npz[Y_key]
        ANGLE = npz[ANGLE_key]
        midline_length = npz[midline_length_key]
        missing = npz[missing_key].astype(bool)
        frame_numbers = npz[frame_key]
    
    return X, Y, missing, ANGLE, midline_length, frame_numbers

import numpy as np

def fill_missing_data_modified(missing_fly_data, distance_threshold):
    missing_data_fixed = 0  # Add a counter for fixed missing data points

    for fly_data in missing_fly_data:
        X = fly_data["X"]
        Y = fly_data["Y"]
        missing = fly_data["missing"]
        ANGLE = fly_data["ANGLE"]
        midline_length = fly_data["midline_length"]
        num_frames = X.shape[0]

        # Find missing data points (where missing == 1)
        missing_indices = np.where(missing == 1)[0]

        for frame in missing_indices:
            prev_frame = frame - 1
    
            if prev_frame < 0:
                continue

            other_fly_data_filtered = [other_fly_data for other_fly_data in missing_fly_data if other_fly_data["fly_number"] != fly_data["fly_number"] and prev_frame < len(other_fly_data["X"])]

            other_X = np.array([other_fly_data["X"][prev_frame] for other_fly_data in other_fly_data_filtered])
            other_Y = np.array([other_fly_data["Y"][prev_frame] for other_fly_data in other_fly_data_filtered])
            other_missing = np.array([other_fly_data["missing"][prev_frame] for other_fly_data in other_fly_data_filtered])

            # Calculate Euclidean distances between the current fly and other flies in the previous frame
            distances = np.sqrt((X[prev_frame] - other_X)**2 + (Y[prev_frame] - other_Y)**2)
            distances[other_missing == 1] = float('inf')  # Set distances to missing flies to infinity

            if len(distances) > 0:
                min_distance = np.min(distances)
                min_index = np.argmin(distances)
            else:
                min_distance = float('inf')  # Set a default value if the array is empty
                min_index = -1  # Set a default index if the array is empty

            if min_distance > distance_threshold:
                X[frame] = X[prev_frame]
                Y[frame] = Y[prev_frame]
            else:
                if min_index != -1:
                    closest_fly = other_fly_data_filtered[min_index]
                    X[frame] = closest_fly["X"][prev_frame]
                    Y[frame] = closest_fly["Y"][prev_frame]

            ANGLE[frame] = ANGLE[prev_frame]
            midline_length[frame] = midline_length[prev_frame]
            missing[frame] = 0  # Update missing status to 0 (not missing)
            missing_data_fixed += 1


    if missing_data_fixed > 0:
        print(f"Fixed {missing_data_fixed} missing data points")

    return missing_fly_data


def calculate_angle_distance_diff_single(df_group):
    fly_numbers = sorted([int(col.split('_')[1]) for col in df_group.columns if col.startswith('X_')])
    
    angle_diffs = []
    distance_diffs = []

    for fly1, fly2 in combinations(fly_numbers, 2):
        # Calculate angle differences based on the formula from the paper
        X1 = df_group[f'X_{fly1}']
        Y1 = df_group[f'Y_{fly1}']
        X2 = df_group[f'X_{fly2}']
        Y2 = df_group[f'Y_{fly2}']
        ANGLE1 = df_group[f'ANGLE_{fly1}']
        ANGLE2 = df_group[f'ANGLE_{fly2}']

        # Create boolean masks to filter out NaN and inf values
        mask1 = np.isfinite(X1) & np.isfinite(Y1) & np.isfinite(ANGLE1)
        mask2 = np.isfinite(X2) & np.isfinite(Y2) & np.isfinite(ANGLE2)

        X1_filtered = X1[mask1]
        Y1_filtered = Y1[mask1]
        X2_filtered = X2[mask2]
        Y2_filtered = Y2[mask2]
        ANGLE1_filtered = ANGLE1[mask1]
        ANGLE2_filtered = ANGLE2[mask2]

        # Calculate vectors a and b for valid data
        a_x = X1_filtered + np.cos(np.deg2rad(ANGLE1_filtered))
        a_y = Y1_filtered + np.sin(np.deg2rad(ANGLE1_filtered))
        b_x = X2_filtered + np.cos(np.deg2rad(ANGLE2_filtered))
        b_y = Y2_filtered + np.sin(np.deg2rad(ANGLE2_filtered))

        a = np.array([a_x - X1_filtered, a_y - Y1_filtered])
        b = np.array([b_x - X2_filtered, b_y - Y2_filtered])

        # Calculate angle theta for valid data
        dot_product = np.sum(a * b, axis=0)
        norm_a = np.linalg.norm(a, axis=0)
        norm_b = np.linalg.norm(b, axis=0)
        
        theta = np.arccos(dot_product / (norm_a * norm_b))
        
        # Convert radians to degrees and truncate to range [0, 180]
        theta_degrees = np.rad2deg(theta)
        theta_degrees = np.where(theta_degrees > 180, 360 - theta_degrees, theta_degrees)

        # Check if there are valid angle differences to concatenate
        if len(theta_degrees) > 0:
            angle_diff_series = pd.Series(theta_degrees, name=f'angle_diff_{fly1}_{fly2}')  # Convert numpy array to Series
            angle_diffs.append(angle_diff_series)     # Append the Series to the list
            print(f"Fly Pair: {fly1}, {fly2}, Angle Differences: {theta_degrees}")

        # Calculate distance differences
        distance_diff = np.sqrt((X1_filtered - X2_filtered) ** 2 + (Y1_filtered - Y2_filtered) ** 2)
        distance_diff_series = pd.Series(distance_diff, name=f'distance_diff_{fly1}_{fly2}')
        distance_diffs.append(distance_diff_series)

    # Concatenate the Series in the angle_diffs list
    if angle_diffs:
        angle_diff_df = pd.concat(angle_diffs, axis=1)

        # Combine angle and distance differences
        angle_diff_df = pd.concat(angle_diffs, axis=1)
        distance_diff_df = pd.concat(distance_diffs, axis=1)

        # Concatenate the angle and distance difference DataFrames with the original DataFrame
        df_group = pd.concat([df_group, angle_diff_df, distance_diff_df], axis=1)

    return df_group



# Modified analyze_social_interactions function for DataFrame grouped by sex, genotype, and sample number
def analyze_social_interactions(df_group, midline_length_thresh, angle_diff_thresh):
    fly_numbers = sorted(set(int(col.split('_')[1]) for col in df_group.columns if col.startswith('X_')))
    interactions = []
    print("Running analysis of social interactions, please wait")
    
    for i, j in combinations(fly_numbers, 2):
        # Calculate inter-individual distances for valid data
        distance = df_group[f'distance_diff_{i}_{j}']
        
        # Create boolean masks to filter out NaN and inf values
        mask_distance = np.isfinite(distance)

        distance = distance[mask_distance]  # Apply the mask

        # Calculate angle differences for valid data
        angle_diff = df_group[f'angle_diff_{i}_{j}']
        
        # Create boolean masks to filter out NaN and inf values
        mask_angle_diff = np.isfinite(angle_diff)

        angle_diff = angle_diff[mask_angle_diff]  # Apply the mask

        # Calculate interactions using valid data
        interaction_condition = (distance < midline_length_thresh) & (-angle_diff_thresh < angle_diff) & (angle_diff < angle_diff_thresh)
        
        interaction_groups = (interaction_condition != interaction_condition.shift()).cumsum()
        interaction_groups = interaction_groups[interaction_condition]
        
        interaction_counts = interaction_groups.value_counts()
        long_interactions = interaction_counts[interaction_counts > 45]
        
        for group_index, interacting_frames in long_interactions.items():
            interaction_start = interaction_groups[interaction_groups == group_index].index.min()
            
            # Calculate the total distance traveled by each fly in the 30 frames before the interaction start
            pre_interaction_frames = df_group.loc[interaction_start - 30:interaction_start - 1]
            distance_fly1 = pre_interaction_frames[f'distance_diff_{i}_{j}'].sum()
            distance_fly2 = pre_interaction_frames[f'distance_diff_{j}_{i}'].sum()
            
            # Compare the distances to determine which fly is the initiator
            if distance_fly1 > distance_fly2:
                initiator = i
            elif distance_fly1 < distance_fly2:
                initiator = j
            else:
                initiator = None
            
            interactions.append((i, j, interaction_start, interacting_frames, initiator))

    interactions_df = pd.DataFrame(interactions, columns=['Fly1', 'Fly2', 'starting_frame', 'duration', 'initiator'])
    return interactions_df



def calculate_inter_individual_distances(df):
    fly_numbers = sorted(set(int(col.split('_')[1]) for col in df.columns if col.startswith('X_')))
    distances = []

    for i, j in combinations(fly_numbers, 2):
        X1 = df[f'X_{i}']
        Y1 = df[f'Y_{i}']
        X2 = df[f'X_{j}']
        Y2 = df[f'Y_{j}']
        
        distance = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)
        distances.append(pd.Series(distance, name=f'distance_{i}_{j}'))

    distances_df = pd.concat([df['frame']] + distances, axis=1)  # Ensure 'frame' column is included

    return distances_df

def process_grouped_output(raw_social_interactions_filepath, output_dir):
    # Load the raw_social_interactions.csv data
    df = pd.read_csv(raw_social_interactions_filepath)
    
    # Compute the total interactions and total duration for each combination of Sex, Genotype, and Sample_Number
    grouped = df.groupby(['Sex', 'Genotype', 'Sample_Number']).agg(
        total_interactions=pd.NamedAgg(column='duration', aggfunc='size'),
        total_duration_frames=pd.NamedAgg(column='duration', aggfunc='sum')
    ).reset_index()
    
    # Convert duration from frames to seconds
    grouped['total_duration_seconds'] = grouped['total_duration_frames'] / 30
    
    # Save the result to socialinteraction_output.csv
    output_filepath = os.path.join(output_dir, "social_interaction_output.csv")
    grouped.to_csv(output_filepath, index=False)
    print(f"Grouped output saved as {output_filepath}")

    return grouped


# Main code that analyzes TREX output containing any amount of flies over time 

def main_modified():
    # Function to open a folder selection dialog
    def select_folder():
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        return folder_path

    # Use the function to get the folder path and find files
    data_dir = select_folder()
    data_files = [filename for filename in os.listdir(data_dir) if filename.endswith(".npz") or filename.endswith(".csv")]  

    for filename in data_files:
    	# Extract the sex, genotype, sample number, and fly number from the filename
        match = re.search(r"^(\w+)_(m|f)_(\d+)_fish(\d+)", filename, re.IGNORECASE)
        if match is None:
            print(f"Warning: {filename} does not match the expected naming convention.")
            continue
        genotype, sex, sample_number, fly_number = match.groups()
        group = (genotype, sex, int(sample_number))
        print(f"Filename: {filename}, Group: {group}, Fly Number: {fly_number}")
 
    
    # Define output directory
    output_dir = os.path.join(data_dir, 'output')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize DataFrames to store all angles, distances and average distances
    avg_distances_by_group_df = pd.DataFrame(columns=['Genotype', 'Sex', 'Sample_Number', 'Average_Distance'])
    
    all_distances_df = pd.DataFrame()
    all_angles_distances_df = pd.DataFrame()

    # Dictionary to store the data for each group (sex, genotype, sample number)
    df_group = defaultdict(list)
    missing_fly_data = []
    pixel_cm_ratio = 0.0042

    for filename in data_files:
        match = re.search(r"^(\w+)_(m|f)_(\d+)_fish(\d+)", filename, re.IGNORECASE)
    
        if match is not None:
            genotype, sex, sample_number, fly_number = match.groups()
            group = (genotype, sex, int(sample_number))
            print(f"Filename: {filename}, Group: {group}, Fly Number: {fly_number}")
            
            # Rest of the processing for the filename
        else:
            print(f"Warning: {filename} does not match the expected naming convention.")
            continue  # This continue statement should be inside the loop


        # Load coordinate data from the file and list missing values
        file_path = os.path.join(data_dir, filename)
        
        if filename.endswith(".npz"):
            X, Y, missing, ANGLE, midline_length, frame_numbers = read_npz_file(file_path)
        elif filename.endswith(".csv"):
            X, Y, missing, ANGLE, midline_length, frame_numbers = read_csv_file(file_path)
        else:
            continue  # Skip this file if it's not an NPZ or CSV file

        # Convert ANGLE to degrees and take the absolute value
        ANGLE = abs(np.degrees(ANGLE))

        # Convert the midline_length to cm
        midline_length *= pixel_cm_ratio

        # Append to dictionary that stores the data for each group and fly
        missing_fly_data.append({"group": group, "fly_number": int(fly_number), "X": X, "Y": Y, "missing": missing, "ANGLE": ANGLE, "midline_length": midline_length, "frame_numbers": frame_numbers})

    # Set the distance threshold in centimeter
    distance_threshold = 1

    # Fix missing data
    print(f"Starting to fix missing data points....")
    fill_missing_data_modified(missing_fly_data, distance_threshold)

    # Organize data by group and fly number
    for fly_data in missing_fly_data:
        group = fly_data["group"]
        fly_number = fly_data["fly_number"]
        X = fly_data["X"]
        Y = fly_data["Y"]
        ANGLE = fly_data["ANGLE"]
        midline_length = fly_data["midline_length"]
        frame_numbers = fly_data["frame_numbers"]

        fly_df = pd.DataFrame({
            'frame': frame_numbers,
            f'X_{fly_number}': X,
            f'Y_{fly_number}': Y,
            f'ANGLE_{fly_number}': ANGLE,
            f'midline_length_{fly_number}': midline_length,
        })

        df_group[group].append(fly_df)

    # Merge the DataFrames by group
    for group, dfs in df_group.items():
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='frame', how='outer')

        # Reorder columns
        column_order = ['frame']
        column_order.extend([col for df in dfs for col in df.columns if 'X_' in col or 'Y_' in col])
        column_order.extend([col for df in dfs for col in df.columns if 'ANGLE_' in col])
        column_order.extend([col for df in dfs for col in df.columns if 'midline_length_' in col])
        merged_df = merged_df[column_order]

        df_group[group] = merged_df
    


    # Calculate angle differences and distances for each group
    for group, df in df_group.items():
        df_group[group] = calculate_angle_distance_diff_single(df)
    
        # Add columns for sex, genotype, and sample number
        df['Sex'], df['Genotype'], df['Sample_Number'] = group
    
        # Append the data of the current group to the all_angles_distances_df
        all_angles_distances_df = pd.concat([all_angles_distances_df, df], ignore_index=True)

    # Save the aggregated angles and distances across all groups to a single CSV file
    output_file_angles_distances = os.path.join(output_dir, "all_angles_distances.csv")
    all_angles_distances_df.to_csv(output_file_angles_distances, index=False)
    print(f"All angles and distances saved as {output_file_angles_distances}")

    
    
    # Calculate distances for each group
    for group, df in df_group.items():

        # Calculate inter-individual distances
        distances_df = calculate_inter_individual_distances(df)
        distances_df['Genotype'], distances_df['Sex'], distances_df['Sample_Number'] = group
        
        print("\nDistances for group:", group)
        print(distances_df.head())
        
        all_distances_df = pd.concat([all_distances_df, distances_df], ignore_index=True)
    
        # Calculate the average distance for the current group
        avg_distance = distances_df.drop(columns=['frame', 'Genotype', 'Sex', 'Sample_Number']).mean().mean()
        print("\nAverage distance for group:", group)
        print(avg_distance)

        # Append the average distance and group details to the avg_distances_by_group_df
        new_row = {
            'Genotype': group[0],
            'Sex': group[1],
            'Sample_Number': group[2],
            'Average_Distance': avg_distance
        }
        avg_distances_by_group_df.loc[len(avg_distances_by_group_df)] = new_row

    # Save the distances between all fly pairs of all groups to a single CSV file
    all_distances_df.to_csv(os.path.join(output_dir, "all_distances.csv"), index=False)

    # Save the average inter individual distances for all groups to a CSV file
    avg_distances_by_group_df.to_csv(os.path.join(output_dir, "average_iid_by_group.csv"), index=False)

    # Define midline length and angle difference thresholds
    midline_length_thresh = 2 * 0.25 
    angle_diff_thresh = 90
    
    # Initialize an empty DataFrame to store interactions from all groups
    all_interactions_df = pd.DataFrame()

    # Calculate and save social interactions for each group
    for group, df in df_group.items():
        print(f"Processing interactions for group: {group}")  # Debugging print statement
        interactions_df = analyze_social_interactions(df, midline_length_thresh, angle_diff_thresh)

        # Add columns for sex, genotype, and sample number
        interactions_df['Sex'], interactions_df['Genotype'], interactions_df['Sample_Number'] = group
	 
        # Append the interactions of the current group to the all_interactions_df
        all_interactions_df = pd.concat([all_interactions_df, interactions_df], ignore_index=True)
	

    # Save the aggregated interactions across all groups as a single CSV file
    output_file = os.path.join(output_dir, f"raw_social_interactions.csv")
    all_interactions_df.to_csv(output_file, index=False)
    process_grouped_output(output_file, output_dir)
    print(f"All interactions saved as {output_file}")



# Call the main function to execute the script
if __name__ == "__main__":
    main_modified()


