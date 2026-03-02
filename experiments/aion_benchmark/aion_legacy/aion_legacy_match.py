import h5py
import pandas as pd
import numpy as np

# File Paths
input_path = '/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5'
csv_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/util_notebooks/leagcy_train_overlap_df.csv'
output_path = 'extracted_matches_small.h5' # <--- New output file


# 1. Load the CSV and convert TARGETID to a set for O(1) fast lookup
df = pd.read_csv(csv_path)
# target_ids_set = set(df['TARGETID'].values)
target_ids_set = df['TARGETID'].unique()[:512]

print(f"Loaded {len(target_ids_set)} IDs from CSV.")

# Lists to collect our extracted data
extracted_abs_indices = []
extracted_target_ids = []
extracted_images_hsc = []
extracted_images_legacy = []

with h5py.File(input_path, 'r') as f:
    print("Keys in input file:", list(f.keys()))

    # Load source types to find where they equal 0
    sources = f['source_type'][:]

    # 2. Get the ABSOLUTE indices where source_type is 0
    # valid_indices contains the actual row numbers in the file
    valid_indices = np.where(sources == 0)[0]

    print(f"Found {len(valid_indices)} objects with source_type == 0. Scanning for matches...")

    # We iterate through the valid indices directly
    # This allows us to access the specific row for IDs and Images
    for count, abs_idx in enumerate(valid_indices):

        # Get the ID at this absolute index
        # We access [abs_idx] directly because object_id_legacy is 1:1 with source_type
        raw_id = f['object_id_legacy'][abs_idx]

        current_id = str(raw_id)[4:][:-2]

        # 3. Check for match
        if current_id in target_ids_set:

            # Save the Absolute Index
            extracted_abs_indices.append(abs_idx)

            # Save the Target ID
            extracted_target_ids.append(current_id)

            # Extract the images using the absolute index
            extracted_images_hsc.append(f['images_hsc'][abs_idx])
            extracted_images_legacy.append(f['images_legacy'][abs_idx])

    print(f"Extraction complete. Found {len(extracted_abs_indices)} matches.")

# 4. Save all extracted data to a new HDF5 file
# We use HDF5 because it handles image arrays much better than CSV/Text files
# h5py does not support numpy Unicode (dtype '<U...'); encode strings as bytes
if len(extracted_abs_indices) > 0:
    target_id_bytes = np.array(
        [s.encode("utf-8") for s in extracted_target_ids],
        dtype=f"S{max(len(s) for s in extracted_target_ids)}",
    )
    with h5py.File(output_path, 'w') as out_f:
        out_f.create_dataset('abs_index', data=np.array(extracted_abs_indices))
        out_f.create_dataset('target_id', data=target_id_bytes)

        # Save images as chunks
        out_f.create_dataset('images_hsc', data=np.array(extracted_images_hsc))
        out_f.create_dataset('images_legacy', data=np.array(extracted_images_legacy))

    print(f"Data successfully saved to {output_path}")
else:
    print("No matches found. Nothing saved.")
