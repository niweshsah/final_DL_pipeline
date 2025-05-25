import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

def run_label_masking_pipeline(
    image_folder: str,
    label_folder: str,
    output_label_folder: str,
    output_image_folder: str,
    label_values=(1, 2, 3),
    dilation_mm=10
):
    """
    Apply binary dilation on segmentation labels and mask the corresponding images.

    Parameters:
        image_folder (str): Path to the original image folder.
        label_folder (str): Path to the predicted segmentation (label) folder.
        output_label_folder (str): Output directory for the modified segmentation masks.
        output_image_folder (str): Output directory for the masked images.
        label_values (tuple): Label values to retain before dilation.
        dilation_mm (float): Dilation radius in millimeters.
    """
    os.makedirs(output_label_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    for label_file in os.listdir(label_folder):
        if not label_file.endswith(".nii.gz"):
            continue

        base_name = label_file.replace(".nii.gz", "")
        label_path = os.path.join(label_folder, label_file)
        print(f"🧠 Processing label: {label_file}")

        try:
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            spacing = label_nii.header.get_zooms()
        except Exception as e:
            print(f"❌ Skipping corrupted label file: {label_file} — {e}")
            continue

        # Keep only specific labels
        binary_label = np.isin(label_data, label_values).astype(np.uint8)

        # Calculate dilation radius in voxel space
        dilation_radius_voxels = np.ceil(dilation_mm / np.array(spacing)).astype(int)
        structure = ndi.generate_binary_structure(3, 1)
        dilated = ndi.binary_dilation(binary_label, structure=structure, iterations=int(max(dilation_radius_voxels)))

        modified_label = dilated.astype(np.uint8)
        out_label_path = os.path.join(output_label_folder, f"{base_name}.nii.gz")

        try:
            new_label_nii = nib.Nifti1Image(modified_label, label_nii.affine, label_nii.header)
            nib.save(new_label_nii, out_label_path)
        except Exception as e:
            print(f"❌ Failed to save label: {label_file} — {e}")
            continue

        # Apply mask to each modality
        for modality in ["0000", "0001", "0002", "0003"]:
            image_path = os.path.join(image_folder, f"{base_name}_{modality}.nii.gz")
            if not os.path.exists(image_path):
                print(f"⚠️  Missing image: {image_path}")
                continue

            try:
                image_nii = nib.load(image_path)
                image_data = image_nii.get_fdata()
            except Exception as e:
                print(f"❌ Skipping corrupted image: {image_path} — {e}")
                continue

            masked_image = image_data * (modified_label > 0)
            new_image_nii = nib.Nifti1Image(masked_image, image_nii.affine, image_nii.header)

            out_image_path = os.path.join(output_image_folder, f"{base_name}_{modality}.nii.gz")

            try:
                nib.save(new_image_nii, out_image_path)
            except Exception as e:
                print(f"❌ Failed to save image: {image_path} — {e}")

if __name__ == "__main__":
    # Example usage
    run_label_masking_pipeline(
        image_folder="/home/teaching/output/Testing/images",
        label_folder="/home/teaching/output/Testing/SegMed_threshold",
        output_label_folder="/home/teaching/output/Testing/labels_threshold",
        output_image_folder="/home/teaching/output/Testing/images_threshold"
    )
