import os
import cv2
import shutil
import yaml
import logging
import random
from glob import glob

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def mask_to_yolo_bboxes(mask, class_id=0):
    """
    Convert a binary segmentation mask into YOLO-format bounding boxes.
    Returns: list of [class_id, x_center, y_center, width, height] (normalized).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    bboxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_c = (x + bw / 2) / w
        y_c = (y + bh / 2) / h
        bw_n = bw / w
        bh_n = bh / h
        bboxes.append([class_id, x_c, y_c, bw_n, bh_n])
    return bboxes

def draw_bboxes(image, bboxes, color=(255, 255, 255), thickness=2):
    """
    Draw YOLO-format bounding boxes on the given image.
    """
    h, w, _ = image.shape
    for cls, x, y, bw, bh in bboxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{cls}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def prepare_patient_datasets(output_root="output-shuffled-t1ce-t2w-flair", dataset_root="dataset/patients"):
    os.makedirs(dataset_root, exist_ok=True)

    total_patients = total_images = total_bboxes = 0

    for patient_id in sorted(os.listdir(output_root)):
        comb_dir = os.path.join(output_root, patient_id, "images", "combined")
        seg_dir  = os.path.join(output_root, patient_id, "labels", "seg")

        if not os.path.isdir(comb_dir) or not os.path.isdir(seg_dir):
            continue

        total_patients += 1

        img_out = os.path.join(dataset_root, patient_id, "images")
        lbl_out = os.path.join(dataset_root, patient_id, "labels")
        viz_out = os.path.join(dataset_root, patient_id, "bbox_viz")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        os.makedirs(viz_out, exist_ok=True)

        for fname in sorted(os.listdir(comb_dir)):
            if not fname.endswith(".png"):
                continue

            img_path = os.path.join(comb_dir, fname)
            seg_name = fname.replace("combined", "seg")
            seg_path = os.path.join(seg_dir, seg_name)

            if not os.path.exists(seg_path):
                logging.warning(f"Missing mask for {img_path}, skipping.")
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

            bboxes = mask_to_yolo_bboxes(mask)

            if not bboxes:
                logging.warning(f"No bounding box found for {fname}, skipping save.")
                continue

            bboxes.sort(key=lambda x: x[3] * x[4], reverse=True)
            bboxes = [bboxes[0]]  # take largest bbox

            total_images += 1
            total_bboxes += len(bboxes)

            # Save image and label
            cv2.imwrite(os.path.join(img_out, fname), img)
            with open(os.path.join(lbl_out, fname.replace(".png", ".txt")), "w") as f:
                for cls, x, y, w_, h_ in bboxes:
                    f.write(f"{cls} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

            # Save visualization
            viz_img = draw_bboxes(img.copy(), bboxes)
            cv2.imwrite(os.path.join(viz_out, fname), viz_img)

        logging.info(f"Processed patient: {patient_id}")

    logging.info(f"\nSummary:\n  Patients: {total_patients}\n  Images: {total_images}\n  BBoxes: {total_bboxes}")

def get_image_label_triplets(patients_dir):
    triplets = []
    for pid in sorted(os.listdir(patients_dir)):
        base = os.path.join(patients_dir, pid)
        img_dir = os.path.join(base, "images", "combined")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(base, "images")
        lbl_dir = os.path.join(base, "labels")
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            logging.warning(f"Skipping {pid}: missing images or labels")
            continue

        for img_path in sorted(glob(os.path.join(img_dir, "*.png"))):
            txt_path = os.path.join(lbl_dir, os.path.basename(img_path).replace(".png", ".txt"))
            if os.path.exists(txt_path):
                triplets.append((img_path, txt_path, pid))
            else:
                logging.warning(f"No label file for {img_path}, skipping.")
    logging.info(f"Found {len(triplets)} image-label pairs.")
    return triplets

def split_triplets(triplets, train_ratio=0.7, val_ratio=0.2):
    random.shuffle(triplets)
    train_size = int(len(triplets) * train_ratio)
    val_size = int(len(triplets) * val_ratio)
    train_t = triplets[:train_size]
    val_t = triplets[train_size:train_size+val_size]
    test_t = triplets[train_size+val_size:]
    return train_t, val_t, test_t

def make_yolo_dirs(base):
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def copy_and_rename(triplets, img_out, lbl_out):
    for img_path, txt_path, pid in triplets:
        new_img = f"{pid}__{os.path.basename(img_path)}"
        new_txt = f"{pid}__{os.path.basename(txt_path)}"
        shutil.copy(img_path, os.path.join(img_out, new_img))
        shutil.copy(txt_path, os.path.join(lbl_out, new_txt))

def write_data_yaml(output_path, nc, train_dir, val_dir, test_dir=None, names=["tumor"]):
    data = {
        'train': os.path.abspath(train_dir),
        'val': os.path.abspath(val_dir),
        'nc': nc,
        'names': names
    }
    if test_dir:
        data['test'] = os.path.abspath(test_dir)

    with open(output_path, 'w') as f:
        yaml.dump(data, f)
    logging.info(f"YOLO data.yaml written to {output_path}")

def prepare_yolov8_with_prefix(
    patients_dir="dataset/patients",
    out_dir="dataset/yolov8",
    train_ratio=0.7,
    val_ratio=0.2,
    class_names=["tumor"]):

    try:
        triplets = get_image_label_triplets(patients_dir)
        train_t, val_t, test_t = split_triplets(triplets, train_ratio, val_ratio)
        logging.info(f"Train/Val/Test sizes: {len(train_t)}/{len(val_t)}/{len(test_t)}")

        make_yolo_dirs(out_dir)
        copy_and_rename(train_t, os.path.join(out_dir, "images/train"), os.path.join(out_dir, "labels/train"))
        copy_and_rename(val_t, os.path.join(out_dir, "images/val"), os.path.join(out_dir, "labels/val"))
        copy_and_rename(test_t, os.path.join(out_dir, "images/test"), os.path.join(out_dir, "labels/test"))

        write_data_yaml(
            output_path=os.path.join(out_dir, "data.yaml"),
            nc=len(class_names),
            train_dir=os.path.join(out_dir, "images/train"),
            val_dir=os.path.join(out_dir, "images/val"),
            test_dir=os.path.join(out_dir, "images/test"),
            names=class_names
        )

        logging.info("YOLOv8 dataset is ready with unique filenames!")

    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")

# Usage
if __name__ == "__main__":
    prepare_patient_datasets()
    prepare_yolov8_with_prefix(
        patients_dir="dataset/patients",
        out_dir="dataset/yolov8xl",
        train_ratio=0.7,
        val_ratio=0.2,
        class_names=["tumor"]
    )