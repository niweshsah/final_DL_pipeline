import os
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry
import torch.nn.functional as F
import random

# --- MedSAM model loading ---
MedSAM_CKPT_PATH = "/DATA/medsam_inference/medsam_vit_b.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

# --- Load YOLO model ---
model = YOLO("/DATA/yolo_on_brain/runs/detect/brain_yolov8m3/weights/best.pt")

# --- Root directory ---
root_folder = "/DATA/yolo_on_brain/output-african"

# --- Dice score accumulation ---
total_intersection = 0
total_pixels = 0
num_images = 0

# --- Traverse all subfolders ---
for subdir, _, _ in os.walk(root_folder):
    if "images/combined" not in subdir:
        continue

    image_dir = subdir
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    random.shuffle(image_paths)

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            continue
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = original_rgb.shape

        resized_img = cv2.resize(original_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        resized_img_tensor = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        img_embed = medsam_model.image_encoder(resized_img_tensor)

        results = model(image_path)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()

        scale_x, scale_y = 1024 / W, 1024 / H
        box_1024_list = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in pred_boxes]

        masks = [medsam_inference(medsam_model, img_embed, [box_1024], H, W) for box_1024 in box_1024_list]

        if masks:
            combined_mask = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)
        else:
            combined_mask = np.zeros((H, W), dtype=np.uint8)

        # --- Ground truth mask ---
        label_seg_dir = image_dir.replace("images/combined", "labels/seg")
        slice_idx = os.path.splitext(image_name)[0].split('_')[-1]
        label_seg_path = os.path.join(label_seg_dir, f"seg_slice_{slice_idx}.png")

        if not os.path.exists(label_seg_path):
            continue

        gt_mask = cv2.imread(label_seg_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        gt_area = np.sum(gt_mask > 0)
        if gt_area == 0:
            continue

        pred_binary = (combined_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        intersection = np.sum(pred_binary * gt_binary)
        A = np.sum(pred_binary)
        B = np.sum(gt_binary)

        total_intersection += intersection
        total_pixels += (A + B)
        num_images += 1

# --- Final Global Dice score ---
if total_pixels > 0:
    global_dice = (2. * total_intersection) / (total_pixels + 1e-8)
    print(f"\n✅ Global Dice score across all images: {global_dice:.4f} using {num_images} images")
else:
    print("❌ No valid tumor regions found in any ground truth masks.")
