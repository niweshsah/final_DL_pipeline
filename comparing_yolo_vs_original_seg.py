import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import json

# === Configure Logging ===
log_filename = "evaluation_log_less_threshold.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === PATHS ===
reference_root = "/DATA/yolo_on_brain/african_total_dataset"
predicted_root = "/DATA/yolo_on_brain/african_medsam_yolo_predicted_nifti"
output_csv = "evaluation_results.csv"
output_json = "evaluation_results_.json"

# === Dice Calculation ===
def dice_coefficient(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    return 2. * intersection / (a.sum() + b.sum() + 1e-8)

# === Evaluation ===
results = []
case_ids = sorted([cid for cid in os.listdir(predicted_root) if cid.startswith("BraTS-SSA")])

for case_id in tqdm(case_ids, desc="Evaluating cases"):
    ref_path = os.path.join(reference_root, case_id, f"{case_id}-seg.nii.gz")
    pred_path = os.path.join(predicted_root, case_id, "seg.nii.gz")
    print(f"🔍 Evaluating: {case_id}")

    if not os.path.exists(ref_path):
        logging.warning(f"❌ Missing reference file for {case_id}")
        continue
    if not os.path.exists(pred_path):
        logging.warning(f"❌ Missing predicted file for {case_id}")
        continue

    try:
        ref_img = nib.load(ref_path)
        pred_img = nib.load(pred_path)

        ref_data = ref_img.get_fdata()
        pred_data = pred_img.get_fdata()

        if ref_data.shape != pred_data.shape:
            pred_data = np.transpose(pred_data, (1, 2, 0))
            if ref_data.shape != pred_data.shape:
                logging.error(f"❌ Shape mismatch in {case_id} even after transpose")
                continue

        ref_bin = (ref_data != 0).astype(np.uint8)
        pred_bin = (pred_data != 0).astype(np.uint8)

        dice = dice_coefficient(ref_bin, pred_bin)
        diff_voxels = np.sum(ref_bin != pred_bin)
        total_voxels = ref_bin.size

        fp = np.sum((pred_bin == 1) & (ref_bin == 0))
        fn = np.sum((pred_bin == 0) & (ref_bin == 1))
        tp = np.sum((pred_bin == 1) & (ref_bin == 1))
        tn = np.sum((pred_bin == 0) & (ref_bin == 0))
        total_tumor_voxels = np.sum(ref_bin)

        results.append({
            "Case": case_id,
            "Dice": dice,
            "Different Voxels": diff_voxels,
            "Total Voxels": total_voxels,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn,
            "FP (%)": 100 * fp / total_tumor_voxels if total_tumor_voxels else 0,
            "FN (%)": 100 * fn / total_tumor_voxels if total_tumor_voxels else 0,
            "TP (%)": 100 * tp / total_tumor_voxels if total_tumor_voxels else 0
        })

        logging.info(f"[{case_id}] ✅ Dice: {dice:.4f} | TP: {tp}, FP: {fp}, FN: {fn}")

    except Exception as e:
        logging.exception(f"⚠️ Error processing {case_id}: {str(e)}")
        continue

# === Save Results to CSV ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n📁 Evaluation complete. Results saved to: {output_csv}")
print(f"📝 Log saved to: {log_filename}")

# === Save Results to JSON (with native Python types) ===
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# === Save Results to JSON (with native Python types), sorted by FN (%) descending ===
results_sorted = sorted(results, key=lambda x: x["FN (%)"])

with open(output_json, "w") as json_file:
    json.dump([ {k: convert_numpy(v) for k, v in item.items()} for item in results_sorted ], json_file, indent=4)
print(f"📁 Sorted results saved to JSON: {output_json}")


# with open(output_json, "w") as json_file:
#     json.dump([ {k: convert_numpy(v) for k, v in item.items()} for item in results ], json_file, indent=4)
# print(f"📁 Results also saved to JSON: {output_json}")

# === Load for Analysis ===
df = pd.read_csv(output_csv)

# === Dice Stats ===
print("\n🔬 Dice Score Statistics:")
print(f"→ Mean Dice:    {df['Dice'].mean():.4f}")
print(f"→ Median Dice:  {df['Dice'].median():.4f}")
print(f"→ Std Dev Dice: {df['Dice'].std():.4f}")
print(f"→ Min Dice:     {df['Dice'].min():.4f}")
print(f"→ Max Dice:     {df['Dice'].max():.4f}")

# === FN% and FP% Stats ===
print("\n🚨 False Negative (FN%) Statistics:")
print(f"→ Mean FN%:     {df['FN (%)'].mean():.2f}%")
print(f"→ Median FN%:   {df['FN (%)'].median():.2f}%")
print(f"→ Std Dev FN:   {df['FN (%)'].std():.2f}")
print(f"→ Max FN%:      {df['FN (%)'].max():.2f}%")
print(f"→ Min FN%:      {df['FN (%)'].min():.2f}%")

print("\n⚠️ False Positive (FP%) Statistics:")
print(f"→ Mean FP%:     {df['FP (%)'].mean():.2f}%")
print(f"→ Median FP%:   {df['FP (%)'].median():.2f}%")
print(f"→ Max FP%:      {df['FP (%)'].max():.2f}%")
print(f"→ Min FP%:      {df['FP (%)'].min():.2f}%")

# === Best & Worst Dice ===
best_case = df.loc[df["Dice"].idxmax()]
worst_case = df.loc[df["Dice"].idxmin()]
print("\n🏆 Best Performing Case:")
print(best_case[["Case", "Dice", "TP", "FP", "FN", "FN (%)", "FP (%)"]])
print("\n💀 Worst Performing Case:")
print(worst_case[["Case", "Dice", "TP", "FP", "FN", "FN (%)", "FP (%)"]])

# === Top 5 Worst FN% ===
print("\n🔥 Top 5 Worst FN% Cases:")
print(df.sort_values(by="FN (%)", ascending=False)[["Case", "Dice", "FN (%)", "FP (%)", "TP", "FP", "FN"]].head(5))

# === Dice Distribution ===
df["Dice Bin"] = pd.cut(df["Dice"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"])
print("\n📊 Dice Score Distribution:")
print(df["Dice Bin"].value_counts().sort_index())

# === FN% Distribution Plot ===
fn_bins = [0, 10, 25, 50, 75, 100]
fn_labels = ["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]
df["FN (%) Bin"] = pd.cut(df["FN (%)"], bins=fn_bins, labels=fn_labels, include_lowest=True)
fn_dist_counts = df["FN (%) Bin"].value_counts().sort_index()

print("\n📈 FN% Distribution:")
print(fn_dist_counts)

plt.figure(figsize=(8, 5))
fn_dist_counts.plot(kind='bar', color='tomato', edgecolor='black')
plt.title("False Negative Percentage (FN%) Distribution")
plt.xlabel("FN% Range")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
