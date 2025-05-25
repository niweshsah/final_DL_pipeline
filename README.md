
# TumourGen

A comprehensive deep learning pipeline for brain tumor analysis, integrating the strengths of **YOLOv8**, **MedSAM**, and **nnU-Net**. TumourGen supports both 2D object detection and 3D pixel-level segmentation for multi-modal medical imaging—tailored for clinical and research applications.

## Pipeline Overview

TumourGen combines three state-of-the-art deep learning architectures to provide a complete solution for brain tumor analysis. The system processes multi-modal medical imaging data through a sequential workflow that leverages the unique strengths of each component.

## Folder Structure

```
TumourGen/
├── dataset/                   # Datasets for training and inference
├── evaluation/                # Evaluation scripts for segmentation results
├── inference/
│   ├── fast_inference.py      # Inference using precomputed intermediate files
│   ├── full_final_inference.py# Complete inference from start to end
│   ├── masking_before_nnUnet.py# Generate modalities from MedSAM+YOLO masks
│   ├── rgb_stacking.py        # Stack T1ce, T2, and FLAIR as RGB images
│   └── seg_file_using_medsam.py# Segmentation using MedSAM + YOLO
├── training/
│   ├── medsam_finetune/       # MedSAM fine-tuning files and script
│   └── yolo_finetune/         # YOLO preprocessing and fine-tuning
```

## Core Components

| Module      | Description                                |
| ----------- | ------------------------------------------ |
| **YOLOv8**  | 2D tumor detection using bounding boxes    |
| **MedSAM**  | Fine-tuned SAM variant for 2D segmentation |
| **nnU-Net** | Automated 3D tumor segmentation            |

## Environment Setup

### Create Conda Environment

```bash
conda create -n tumor_seg python=3.10
conda activate tumor_seg

# PyTorch with CUDA
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Additional dependencies
conda install -c conda-forge nibabel
pip install -r requirements.txt
```

## 2D Tumor Detection with YOLOv8

### Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── brats_yolo.yaml
```

**`brats_yolo.yaml`**:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ["tumor"]
```

### Fine-Tuning YOLOv8

```bash
yolo task=detect mode=train model=yolov8m.pt data=brats_yolo.yaml epochs=100 imgsz=640 batch=16 device=0
```

## MedSAM Fine-Tuning (with LoRA)

### Why LoRA?

**LoRA (Low-Rank Adaptation)** reduces training time and memory by inserting trainable low-rank matrices into frozen transformer layers—ideal for large models like MedSAM.

### Fine-Tuning Script

```bash
python lora_fine_tune.py \
  --img_folder /path/to/images \
  --mask_folder /path/to/masks \
  --train_img_list /path/to/train.txt \
  --val_img_list /path/to/val.txt \
  --sam_ckpt /path/to/sam_vit_b.pth \
  --dir_checkpoint ./checkpoints/lora_medsam \
  --epochs 50 \
  --b 4 \
  --lr 1e-4 \
  --num_cls 2 \
  --targets 1 \
  --arch vit_b \
  --finetune_type lora \
  --if_warmup
```

> **Note:** Ensure paths and preprocessing logic are properly configured in the script.

## 3D Tumor Segmentation with nnU-Net

### Installation

```bash
pip install nnunet
```

### Set Environment Variables

```bash
echo 'export nnUNet_raw_data_base="/path/to/data/nnUNet_raw_data_base"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/path/to/data/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export RESULTS_FOLDER="/path/to/nnUNet_trained_models"' >> ~/.bashrc
source ~/.bashrc
```

### Dataset Organization

```
nnUNet_raw_data_base/
└── nnUNet_raw_data/
    └── TaskXXX_MYTASK/
        ├── imagesTr/
        ├── labelsTr/
        ├── imagesTs/
        ├── labelsTs/
        └── dataset.json
```

### Modality Mapping

| Modality    | Filename Suffix |
| ----------- | --------------- |
| T1-native   | `_0000.nii.gz`  |
| T1-contrast | `_0001.nii.gz`  |
| T2          | `_0002.nii.gz`  |
| T2-FLAIR    | `_0003.nii.gz`  |

### Label Definitions

| Label | Class                                   |
| ----- | --------------------------------------- |
| 0     | Background                              |
| 1     | Necrotic/Non-enhancing Tumor Core (NCR) |
| 2     | Edematous/Invaded Tissue (ED)           |
| 3     | Enhancing Tumor (ET)                    |

## Running the nnU-Net Pipeline

### Training

```bash
python train.py \
  --task_number 102 \
  --task_name Task102_BratsMix \
  --fold 0 \
  --configuration 3d_fullres \
  --trainer_class nnUNetTrainerV2
```

### Inference

```bash
python test.py --root_dir /path/to/data/Test_Ped
```

### Evaluation

```bash
python evaluate.py \
  -ref /path/to/data/labelsTs \
  -pred /path/to/output \
  -l 1 2 3
```

## Sample Test Dataset Layout

```
test_data/
├── imagesTs/
└── labelsTs/
```

## Public Datasets

| Dataset                        | Link                                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **BraTS 2023 (Adult Glioma)**  | [Part 1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) · [Part 2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip) |
| **BraTS-SSA (Sub-Saharan)**    | [Download](https://www.kaggle.com/datasets/mrasiamah/brats2023-ssa)                                                                      |
| **BraTS-PED 2024 (Pediatric)** | [Download](https://www.kaggle.com/datasets/srutorshibasuray/brats-ped-2024)                                                              |

## Contributing

We welcome contributions to TumourGen! Please read our contributing guidelines and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TumourGen in your research, please cite:

```bibtex
@misc{tumourgen2024,
  title={TumourGen: A Comprehensive Deep Learning Pipeline for Brain Tumor Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/TumourGen}}
}
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [MedSAM GitHub Repository](https://github.com/bowang-lab/MedSAM)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [nnU-Net GitHub Repository](https://github.com/MIC-DKFZ/nnUNet)
- [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1904.08128)
