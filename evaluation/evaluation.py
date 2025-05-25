import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
import argparse

# === Command line arguments ===
parser = argparse.ArgumentParser(description="Evaluate multi-label segmentation results with Dice score.")
parser.add_argument("-ref", "--reference_dir", required=True, help="Path to reference NIfTI folder")
parser.add_argument("-pred", "--predicted_dir", required=True, help="Path to predicted NIfTI folder")
parser.add_argument("-l", "--labels", nargs="+", type=int, required=True, help="List of integer labels to evaluate")
args = parser.parse_args()

reference_dir = args.reference_dir
predicted_dir = args.predicted_dir
labels = args.labels

# === Configure Logging ===
log_filename = os.path.join(predicted_dir, "evaluation_multilabel.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Output paths (inside predicted_dir)
output_csv = os.path.join(predicted_dir, "multilabel_evaluation.csv")
output_json = os.path.join(predicted_dir, "multilabel_evaluation.json")

# Get sorted case IDs from predicted_dir, filtering for folders starting with "BraTS-SSA"
case_ids = sorted([cid for cid in os.listdir(predicted_dir) if cid.startswith("BraTS-SSA")])

def dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum() + 1e-8)

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

        # Evaluate each label separately
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

        # Evaluate whole tumor (any of the labels)
        ref_whole = np.isin(ref_data, labels)
        pred_whole = np.isin(pred_data, labels)

        dice_whole = dice_score(ref_whole, pred_whole)
        tp_whole = np.sum(pred_whole & ref_whole)
        fp_whole = np.sum(pred_whole & ~ref_whole)
        fn_whole = np.sum(~pred_whole & ref_whole)
        tn_whole = np.sum((~pred_whole) & (~ref_whole))
        total_voxels_whole = np.sum(ref_whole)

        results.append({
            "Case": case_id,
            "Label": "WholeTumor",
            "Dice": dice_whole,
            "TP": tp_whole,
            "FP": fp_whole,
            "FN": fn_whole,
            "TN": tn_whole,
            "FP (%)": 100 * fp_whole / (total_voxels_whole + 1e-8),
            "FN (%)": 100 * fn_whole / (total_voxels_whole + 1e-8),
            "TP (%)": 100 * tp_whole / (total_voxels_whole + 1e-8)
        })

        logging.info(f"[{case_id}] Label WholeTumor ✅ Dice: {dice_whole:.4f} | TP: {tp_whole}, FP: {fp_whole}, FN: {fn_whole}")

    except Exception as e:
        logging.exception(f"⚠️ Error processing {case_id}: {str(e)}")

# Save CSV and JSON
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
    json.dump([{k: convert_numpy(v) for k, v in row.items()} for row in results], f, indent=4)

print(f"\n📁 Evaluation complete. CSV: {output_csv}, JSON: {output_json}")
