import os
import numpy as np
import nibabel as nib
import cv2
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def normalize_and_convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0–255 and convert to uint8.
    """
    if np.max(img) == np.min(img):
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255
    return img.astype(np.uint8)

def resolve_nifti_path(path: str) -> str | None:
    """
    If `path` is a directory, return the first NIfTI file (.nii or .nii.gz) found inside.
    If it's a file, return it if valid.
    """
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(('.nii', '.nii.gz')):
                full_path = os.path.join(path, f)
                if os.path.isfile(full_path):
                    return full_path
    elif os.path.isfile(path):
        return path
    return None

def find_nifti_file(patient_path: str, suffixes: tuple) -> str | None:
    """
    Find the file in `patient_path` that ends with one of the given `suffixes`.
    """
    for f in os.listdir(patient_path):
        if f.endswith(suffixes):
            full_path = os.path.join(patient_path, f)
            return resolve_nifti_path(full_path)
    return None

def save_3channel_image(flair_path, t2_path, t1ce_path, seg_path, patient_id, axis=2, output_folder="output-shuffled"):
    """
    Save 2D slices of 3-channel (flair, t2, t1ce) images and binary segmentation masks as PNGs.
    """

    try:
        # Load volumes
        flair_img = nib.load(flair_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()
        t1ce_img = nib.load(t1ce_path).get_fdata()
        seg_img = nib.load(seg_path).get_fdata()
    except Exception as e:
        logging.error(f"Skipping {patient_id}: Failed to load one or more NIfTI files. Reason: {e}")
        return

    # Normalize intensities
    flair_img = normalize_and_convert_to_uint8(flair_img)
    t2_img = normalize_and_convert_to_uint8(t2_img)
    t1ce_img = normalize_and_convert_to_uint8(t1ce_img)

    # Convert segmentation to binary mask
    seg_img = (seg_img > 0).astype(np.uint8) * 255

    # Create output directories
    patient_folder = os.path.join(os.path.abspath(output_folder), patient_id)
    image_dirs = {
        "flair": os.path.join(patient_folder, "images", "flair"),
        "t2": os.path.join(patient_folder, "images", "t2"),
        "t1ce": os.path.join(patient_folder, "images", "t1ce"),
        "combined": os.path.join(patient_folder, "images", "combined"),
        "seg": os.path.join(patient_folder, "labels", "seg")
    }

    for dir_path in image_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Loop over slices
    num_slices = flair_img.shape[axis]
    for i in range(num_slices):
        # Extract slices
        flair_slice = np.take(flair_img, i, axis)
        t2_slice = np.take(t2_img, i, axis)
        t1ce_slice = np.take(t1ce_img, i, axis)
        seg_slice = np.take(seg_img, i, axis)

        # Create combined RGB image (t1ce - R, t2 - G, flair - B)
        combined_img = cv2.merge([t1ce_slice, t2_slice, flair_slice])

        # Output file paths
        file_name = f"slice_{i:03}.png"
        cv2.imwrite(os.path.join(image_dirs["flair"], f"flair_{file_name}"), flair_slice)
        cv2.imwrite(os.path.join(image_dirs["t2"], f"t2_{file_name}"), t2_slice)
        cv2.imwrite(os.path.join(image_dirs["t1ce"], f"t1ce_{file_name}"), t1ce_slice)
        cv2.imwrite(os.path.join(image_dirs["combined"], f"combined_{file_name}"), combined_img)
        cv2.imwrite(os.path.join(image_dirs["seg"], f"seg_{file_name}"), seg_slice)

    logging.info(f"✅ Processed and saved {num_slices} slices for patient: {patient_id}")

def process_all_cases(base_folder, output_folder="output-shuffled", axis=2):
    """
    Process all patient folders inside `base_folder`.
    Each patient folder should contain required NIfTI files.
    """
    if not os.path.isdir(base_folder):
        logging.error(f"Base folder '{base_folder}' does not exist or is not a directory.")
        return

    patient_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    if not patient_folders:
        logging.warning("No patient folders found.")
        return

    for patient_folder in sorted(patient_folders):
        patient_path = os.path.join(base_folder, patient_folder)

        # Try to locate each modality
        flair_path = find_nifti_file(patient_path, ('-t2f.nii', '-t2f.nii.gz'))
        t2_path = find_nifti_file(patient_path, ('-t2w.nii', '-t2w.nii.gz'))
        t1ce_path = find_nifti_file(patient_path, ('-t1c.nii', '-t1c.nii.gz'))
        seg_path = find_nifti_file(patient_path, ('-seg.nii', '-seg.nii.gz'))

        if all([flair_path, t2_path, t1ce_path, seg_path]):
            save_3channel_image(flair_path, t2_path, t1ce_path, seg_path, patient_folder, axis, output_folder)
        else:
            logging.warning(f"⚠️ Skipping {patient_folder}: Missing one or more required NIfTI files.")

if __name__ == "__main__":
    # Modify this path according to your dataset location
    base_data_path = "/home/rocinate/Desktop/pediatric"
    output_path = "output-shuffled-t1ce-t2w-flair"

    # Process with axis=2 by default (axial slices)
    process_all_cases(base_data_path, output_folder=output_path, axis=2)
