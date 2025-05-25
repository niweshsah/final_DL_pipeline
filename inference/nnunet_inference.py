import os
import subprocess
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("nnunet_segmentation.log"),
        logging.StreamHandler()
    ]
)


def run_nnunet_inference(
    task_name_or_id: str,
    input_folder: str,
    output_folder: str,
    model: str = "3d_fullres",
    folds: str = "all",
    device: str = "cuda"
):
    """
    Runs nnUNetv2 prediction using CLI for a given task on all images in a folder.
    """
    if not os.path.exists(input_folder):
        logging.error(f"Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    logging.info("Starting nnU-Net inference...")
    try:
        cmd = [
            "nnUNet_predict",
            "-i", input_folder,
            "-o", output_folder,
            "-t", task_name_or_id,
            "-m", model,
            "-f", folds,
            "--disable_tta"
        ]

        logging.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info("nnU-Net inference completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"nnU-Net inference failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during nnU-Net inference: {e}")


if __name__ == "__main__":
    run_nnunet_inference(
        task_name_or_id="Task103_BratsGen",  # or use number like "101"
        input_folder="./Testing/images_new",  # NIfTI files (.nii.gz) here
        output_folder="./final_output",
        model="3d_fullres",
        folds="all",
        device="cuda"
    )