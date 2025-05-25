import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

# 💥 Config
image_folder = "/home/teaching/output/Testing/images"
label_folder = "/home/teaching/output/Testing/SegMed_threshold"
output_label_folder = "/home/teaching/output/Testing/labels_threshold"
output_image_folder = "/home/teaching/output/Testing/images_threshold"

os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

def process_labels_and_images():
    for label_file in os.listdir(label_folder):
        if not label_file.endswith(".nii.gz"):
            continue

        base_name = label_file.replace(".nii.gz", "")  # remove "_seg.nii.gz"
        label_path = os.path.join(label_folder, label_file)
        print(base_name)
        try:
            print(f"🧠 Processing label: {label_file}")
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            spacing = label_nii.header.get_zooms()
        except Exception as e:
            print(f"❌ Skipping corrupted label file: {label_file} — {e}")
            continue

        # Convert label values 1, 2, 3 to 1; others to 0
        binary_label = np.isin(label_data, [1, 2, 3]).astype(np.uint8)

        # Dilation buffer of 5mm in voxel space
        dilation_radius = np.ceil(10 / np.array(spacing)).astype(int)
        structure = ndi.generate_binary_structure(3, 1)
        dilated = ndi.binary_dilation(binary_label, structure=structure, iterations=int(max(dilation_radius)))

        # Final modified label
        modified_label = (dilated > 0).astype(np.uint8)
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
                print(f"⚠️  Missing: {image_path}")
                continue

            try:
                image_nii = nib.load(image_path)
                image_data = image_nii.get_fdata()
            except Exception as e:
                print(f"❌ Skipping corrupted image: {image_path} — {e}")
                continue

            # Apply label mask
            masked_image = image_data * (modified_label > 0)
            new_image_nii = nib.Nifti1Image(masked_image, image_nii.affine, image_nii.header)

            out_image_path = os.path.join(output_image_folder, f"{base_name}_{modality}.nii.gz")

            try:
                nib.save(new_image_nii, out_image_path)
            except Exception as e:
                print(f"❌ Failed to save image: {image_path} — {e}")

if __name__ == "__main__":
    process_labels_and_images()
