import os
import subprocess
import argparse

def preprocess_nnunet(task_number):
    """Preprocess dataset with nnU-Net."""
    print("Preprocessing dataset with nnU-Net...")
    
    cmd = [
        "nnUNet_plan_and_preprocess",  # nn-U-Net preprocessing command
        "-t", task_number,    # Task number (e.g., 102)
        "--verify_dataset_integrity",  # Verify dataset integrity flag
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Preprocessing completed.")

def train_nnunet(configuration, trainer_class, task_name, fold):
    """Train nn-U-Net on a given task and fold using the specified trainer."""
    print("Training nn-U-Net...")
    
    cmd = [
        "nnUNet_train",                # nn-U-Net training command
        configuration,                 # Configuration (e.g., "3d_fullres")
        trainer_class,                 # Trainer class (e.g., "nnUNetTrainerV2")
        task_name,                     # Full task name (e.g., Task102_BratsMix)
        str(fold),                     # Fold number (0-5)
        "--npz"                        # Option to save results as .npz files
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Training completed.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train nnU-Net model.")
    parser.add_argument(
        "--task_number", type=str, required=True, help="The task number for the dataset (e.g., 102)."
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="The full task name for the dataset (e.g., Task102_BratsMix)."
    )
    parser.add_argument(
        "--fold", type=int, required=True, help="The fold to train on (0-5)."
    )
    parser.add_argument(
        "--configuration", type=str, required=True, help="The configuration for nn-U-Net (e.g., '3d_fullres')."
    )
    parser.add_argument(
        "--trainer_class", type=str, required=True, help="The trainer class (e.g., 'nnUNetTrainerV2')."
    )

    return parser.parse_args()

def main():
    # Parse arguments from command line
    args = parse_args()

    # Preprocess dataset using task number
    try:
        preprocess_nnunet(args.task_number)
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return

    # Train the model using the full task name
    try:
        train_nnunet(args.configuration, args.trainer_class, args.task_name, args.fold)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

if __name__ == "__main__":
    main()

# eg Command: python train.py --task_number 102 --task_name Task102_BratsMix --fold 0 --configuration 3d_fullres --trainer_class nnUNetTrainerV2