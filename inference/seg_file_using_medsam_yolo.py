import os
import numpy as np
import torch
import cv2
import nibabel as nib
import torch.nn.functional as F
from ultralytics import YOLO
from segment_anything import sam_model_registry
import logging


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("medsam_yolo_segmentation.log"),
        logging.StreamHandler()
    ]
)

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    try:
        box_torch = torch.tensor(box_1024, dtype=torch.float32, device=img_embed.device)
        if box_torch.ndim == 2:
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        return (low_res_pred > 0.5).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error during MedSAM inference: {e}")
        return np.zeros((H, W), dtype=np.uint8)


def run_medsam_yolo_pipeline(
    device: str,
    yolo_weights_path: str,
    medsam_ckpt_path: str,
    root_folder: str,
    output_root: str,
    buffer_box: int = 10,
    conf_thresh: float = 0.1,
    iou_thresh: float = 0.5,
    yolo_class_filter: list = [0]
):
    # Load Models
    try:
        logging.info("Loading MedSAM model...")
        medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_ckpt_path).to(device)
        medsam_model.eval()
        logging.info("MedSAM loaded successfully.")

        logging.info("Loading YOLO model...")
        yolo_model = YOLO(yolo_weights_path)
        logging.info("YOLO loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return

    # Process each case folder
    for case_folder_name in sorted(os.listdir(root_folder)):
        case_folder = os.path.join(root_folder, case_folder_name)
        if not os.path.isdir(case_folder):
            continue

        image_dir = os.path.join(case_folder, "images/combined")
        if not os.path.exists(image_dir):
            logging.warning(f"No images/combined directory in {case_folder}")
            continue

        try:
            image_paths = sorted([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        except Exception as e:
            logging.warning(f"Could not list files in {image_dir}: {e}")
            continue

        if not image_paths:
            logging.warning(f"No image files found in {image_dir}")
            continue

        pred_masks = []
        logging.info(f"Processing folder: {image_dir}")

        for image_path in image_paths:
            try:
                image_name = os.path.basename(image_path)
                original_bgr = cv2.imread(image_path)
                if original_bgr is None:
                    logging.warning(f"Failed to read image: {image_path}")
                    continue

                original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
                H, W, _ = original_rgb.shape

                resized_img = cv2.resize(original_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                resized_img_tensor = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                img_embed = medsam_model.image_encoder(resized_img_tensor)

                # results = yolo_model(image_path, conf=conf_thresh, iou=iou_thresh, classes=yolo_class_filter)
                results = yolo_model(image_path,classes=yolo_class_filter)
                
                pred_boxes = results[0].boxes.xyxy.cpu().numpy()

                scale_x, scale_y = 1024 / W, 1024 / H
                box_1024_list = []
                for box in pred_boxes:
                    x1, y1, x2, y2 = box
                    x1 = max(x1 - buffer_box, 0)
                    y1 = max(y1 - buffer_box, 0)
                    x2 = min(x2 + buffer_box, W - 1)
                    y2 = min(y2 + buffer_box, H - 1)
                    box_1024 = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                    box_1024_list.append(box_1024)

                masks = [medsam_inference(medsam_model, img_embed, [box_1024], H, W)
                         for box_1024 in box_1024_list]
                combined_mask = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8) if masks else np.zeros((H, W), dtype=np.uint8)

                slice_index = int(image_name.split('_')[-1].split('.')[0])
                pred_masks.append((slice_index, combined_mask))
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {e}")
                continue

        if not pred_masks:
            logging.warning(f"No valid masks generated for {image_dir}")
            continue

        try:
            pred_masks.sort()
            volume = np.stack([mask for _, mask in pred_masks], axis=0)  # (Z, H, W)
            volume = np.transpose(volume, (1, 2, 0))  # (H, W, Z)

            nii_file = os.path.join(case_folder, "labels/nii/seg.nii.gz")
            if os.path.exists(nii_file):
                nii_data = nib.load(nii_file)
                affine, header = nii_data.affine, nii_data.header
            else:
                affine, header = np.eye(4), None
                logging.warning(f"Original seg.nii.gz not found for {case_folder}. Using identity affine.")

            output_case_dir = os.path.join(output_root, case_folder_name)
            os.makedirs(output_case_dir, exist_ok=True)
            output_path = os.path.join(output_case_dir, "seg.nii.gz")

            nifti_img = nib.Nifti1Image(volume, affine=affine, header=header)
            nib.save(nifti_img, output_path)
            logging.info(f"Saved segmentation: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save NIfTI for {image_dir}: {e}")


if __name__ == "__main__":
    run_medsam_yolo_pipeline(
        device="cuda:0",
        yolo_weights_path="/DATA/yolo_on_brain/runs/detect/brain_yolov8/weights/best.pt",
        medsam_ckpt_path="/DATA/medsam_inference/medsam_vit_b.pth",
        root_folder="/DATA/yolo_on_brain/output-african",
        output_root="/DATA/yolo_on_brain/african_medsam_yolo_predicted_nifti_less_threshold",
        buffer_box=10
    )
