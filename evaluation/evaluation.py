import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import json

# === Configure Logging ===
log_filename = "evaluation_multilabel.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === Paths ===
reference_dir = "/path/to/reference_nifti"
predicted_dir = "/path/to/predicted_nifti"
output_csv = "multilabel_evaluation.csv"
output_json = "multilabel_evaluation.json"

# === Parameters ===
labels = [1, 2, 3, 4]  # Define all labels present in the segmentation
case_ids = sorted([cid for cid in os.listdir(predicted_dir) if cid.startswith("BraTS-SSA")])

# === Dice Calculation ===
def dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum() + 1e-8)

# === Evaluation ===
results = []

for case_id in tqdm(case_ids, desc="Evaluating cases"):
    ref_path = os.path.join(reference_dir, case_id, f"{case_id}-seg.nii.gz")
    pred_path = os.path.join(predicted_dir, case_id, "seg.nii.gz")
    print(f"🔍 Evaluating: {case_id}")

    if not os.path.exists(ref_path) or not os.path.exists(pred_path):
        logging.warning(f"❌ Missing data for {case_id}")
        continue

    try:
        ref_data = nib.load(ref_path).get_fdata().astype(np.int16)
        pred_data = nib.load(pred_path).get_fdata().astype(np.int16)

        if ref_data.shape != pred_data.shape:
            pred_data = np.transpose(pred_data, (1, 2, 0))
            if ref_data.shape != pred_data.shape:
                logging.error(f"❌ Shape mismatch in {case_id}")
                continue

        for label in labels:
            ref_bin = (ref_data == label)
            pred_bin = (pred_data == label)

            dice = dice_score(ref_bin, pred_bin)
            tp = np.sum(pred_bin & ref_bin)
            fp = np.sum(pred_bin & ~ref_bin)
            fn = np.sum(~pred_bin & ref_bin)
            tn = np.sum((~pred_bin) & (~ref_bin))

            total_label_voxels = np.sum(ref_bin)

            results.append({
                "Case": case_id,
                "Label": label,
                "Dice": dice,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "FP (%)": 100 * fp / (total_label_voxels + 1e-8),
                "FN (%)": 100 * fn / (total_label_voxels + 1e-8),
                "TP (%)": 100 * tp / (total_label_voxels + 1e-8)
            })

            logging.info(f"[{case_id}] Label {label} ✅ Dice: {dice:.4f} | TP: {tp}, FP: {fp}, FN: {fn}")

    except Exception as e:
        logging.exception(f"⚠️ Error processing {case_id}: {str(e)}")

# === Save CSV and JSON ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(output_json, "w") as f:
    json.dump([ {k: convert_numpy(v) for k, v in row.items()} for row in results ], f, indent=4)

print(f"\n📁 Evaluation complete. CSV: {output_csv}, JSON: {output_json}")
