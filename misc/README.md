
# 🧠 TumourGen

A comprehensive pipeline combining **MedSAM**, **nnU-Net**, and **YOLOv8** for multi-modal brain tumor segmentation and localization. This framework supports both pixel-level segmentation and object-level detection using 3D and 2D medical imaging data.

---

## 🔄 Pipeline Overview

![Pipeline Architecture](path/to/pipeline_image.png)

### 🧩 Components

* **MedSAM**: Interactive/fine-tuned SAM-based 2D segmenter
* **nnU-Net**: 3D segmentation pipeline with preprocessing, training, inference, and evaluation
* **YOLOv8**: 2D tumor detection with bounding boxes

---

## 🌟 Key Features

✅ Pre-trained **MedSAM** and **YOLOv8** models
✅ **LoRA fine-tuning** support for efficient MedSAM adaptation
✅ Fully automated **nnU-Net** training and inference
✅ Supports 3D full-resolution segmentation
✅ **Cross-validation** and **per-class Dice score evaluation**
✅ Easy-to-use **command-line tools**

---

## 📦 Environment Setup

### 🐍 Create Conda Environment

```bash
conda create -n tumor_seg python=3.10
conda activate tumor_seg

# Install PyTorch with GPU support
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Other dependencies
conda install -c conda-forge nibabel
pip install -r requirements.txt
```

### 🧠 Install nnU-Net and Setup Models

```bash
pip install nnunet

# Set up environment variables
echo 'export nnUNet_raw_data_base="/path/to/data/nnUNet_raw_data_base"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/path/to/data/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export RESULTS_FOLDER="/path/to/nnUNet_trained_models"' >> ~/.bashrc
source ~/.bashrc

# Place MedSAM model
cp medsam_vit_b.pth /path/to/MedSAM/

# Extract nnU-Net trained models
unzip nnUNet_trained_models.zip -d $RESULTS_FOLDER
```

---

## 🗂️ Dataset Organization

### 🏋️ For Training (nnU-Net)

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

### 🔍 For Testing

```
test_data/
├── imagesTs/
└── labelsTs/
```

---

## 📊 Available Datasets

### 🧠 Mapping Convention

| Modality          | Filename suffix |
| ----------------- | --------------- |
| T1-native (t1n)   | `_0000.nii.gz`  |
| T1-contrast (t1c) | `_0001.nii.gz`  |
| T2-weighted (t2w) | `_0002.nii.gz`  |
| T2-FLAIR (t2f)    | `_0003.nii.gz`  |

| Label | Meaning                                 |
| ----- | --------------------------------------- |
| 0     | Background                              |
| 1     | Necrotic/Non-enhancing Tumor Core (NCR) |
| 2     | Edematous/Invaded Tissue (ED)           |
| 3     | Enhancing Tumor (ET)                    |

### 🧾 Downloadable Datasets

1. **Adult Glioma (BraTS 2023)**
   [Download Part 1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1)
   [Download Part 2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip)

2. **Sub-Saharan Dataset (BraTS-SSA)**
   [Download](https://www.kaggle.com/datasets/mrasiamah/brats2023-ssa)

3. **Pediatric Dataset (BraTS-PED 2024)**
   [Download](https://www.kaggle.com/datasets/srutorshibasuray/brats-ped-2024)

---

## 🚀 Quick Start

### 1️⃣ Preprocessing

```bash
python preProcess.py --root_dir /path/to/data/SSA
```

### 2️⃣ Training with nnU-Net

```bash
python train.py \
  --task_number 102 \
  --task_name Task102_BratsMix \
  --fold 0 \
  --configuration 3d_fullres \
  --trainer_class nnUNetTrainerV2
```

### 3️⃣ Inference

```bash
python test.py --root_dir /path/to/data/Test_Ped
```

### 4️⃣ Evaluation

```bash
python evaluate.py -ref /path/to/data/labelsTs -pred /path/to/output -l 1 2 3
```

---

## 🎯 YOLOv8 Tumor Detection (2D)

### 📁 Dataset Format

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

**brats\_yolo.yaml**

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ["tumor"]
```

### 🧪 Fine-Tuning Command

```bash
yolo task=detect mode=train model=yolov8n.pt data=brats_yolo.yaml epochs=100 imgsz=640 batch=16 device=0
```

---

## 🧠 Fine-Tuning MedSAM with LoRA (Low-Rank Adaptation)

### 🤔 Why LoRA?

LoRA reduces the number of trainable parameters by injecting low-rank matrices into attention layers. This allows **efficient fine-tuning** of large models like SAM on specialized medical tasks.

### 🔧 How It Works

* LoRA is applied to the **self-attention layers** of the ViT encoder in MedSAM.
* All original model weights are **frozen**.
* Only LoRA layers are trained using brain tumor masks from BraTS.

### 🧪 Fine-Tuning Command

```bash
python finetune_lora_medsam.py \
  --model medsam_vit_b \
  --lora_r 8 \
  --lora_alpha 32 \
  --train_data /path/to/BratsMix/imagesTr \
  --train_labels /path/to/BratsMix/labelsTr \
  --epochs 20 \
  --lr 1e-4 \
  --save_path ./checkpoints/medsam_lora_brats.pth
```

### 📌 Notes

* Requires `peft`, `transformers`, and a LoRA-compatible MedSAM fork.
* Visualization is supported via `notebooks/medsam_visualize.ipynb`.

---

## 📈 Performance Monitoring

We report the following metrics:

* **Dice Score Coefficient (DSC)** for each tumor label
* Fold-wise and overall validation accuracy
* Per-class segmentation performance
* Inference speed and memory footprint

---

## 📚 References & Resources

* [nnU-Net Paper](https://arxiv.org/abs/1904.08128)
* [MedSAM Repository](https://github.com/bowang-lab/MedSAM)
* [Official nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
* [YOLOv8 Docs](https://docs.ultralytics.com)
* [LoRA Paper](https://arxiv.org/abs/2106.09685)

