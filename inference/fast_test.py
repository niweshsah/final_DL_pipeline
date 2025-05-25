import os
import numpy as np
import nibabel as nib
import cv2
import logging
from seg_file_using_medsam_yolo import run_medsam_yolo_pipeline
from masking_before_nnUnet import run_label_masking_pipeline
from nnunet_inference import run_nnunet_inference
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    # Modify this path according to your dataset location
    masked_data_path = "/home/teaching/output/Testing/masked_images"
    output_path = "/home/teaching/output/Testing/predicted"

    run_nnunet_inference(
        task_name_or_id="Task103_BratsGen",  # or use number like "101"
        input_folder="/home/teaching/output/Testing/images_threshold",  # NIfTI files (.nii.gz) here
        output_folder="/home/teaching/output/Testing/predicted",
        model="3d_fullres",
        folds="all",
        device="cuda"
    )


