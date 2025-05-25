import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
input_folder = '/DATA/yolo_on_brain/output-african/BraTS-SSA-00106-000/predicted_masks_npy'     # Replace with your folder
output_folder = 'images_from_npy'

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all .npy files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        # Load the .npy file
        file_path = os.path.join(input_folder, filename)
        matrix = np.load(file_path)

        # Check if the matrix is 2D
        if matrix.ndim != 2:
            print(f"Skipping {filename}: not a 2D matrix")
            continue

        # Convert 0/1 to 0/255 (black/white)
        img_array = (matrix * 255).astype(np.uint8)

        # Save image using matplotlib
        output_path = os.path.join(output_folder, filename.replace('.npy', '.png'))
        plt.imsave(output_path, img_array, cmap='gray')

        print(f"Saved {output_path}")
