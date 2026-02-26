import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

ROOT_DIR = "AI_QUALITY_ENGINEERING"
DEVICES = ["iphone 12", "VIVO Y51A"]

OUTPUT_DIR = r"C:\Users\Lucifer\python_workspace\BITS\AI_Quality_Engineering\dataset"
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

CLASSES = [
    "animal",
    "name_board",
    "vehicle",
    "pedestrian",
    "pothole",
    "road_sign",
    "speed_breaker"
]

random.seed(42)

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================

for split in SPLIT_RATIOS:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ==============================
# COLLECT ALL IMAGE+JSON PAIRS
# ==============================

samples = []
for device in DEVICES:
    device_path = os.path.join(ROOT_DIR, device)

    for root, _, files in os.walk(device_path):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                json_path = img_path.replace(".jpg", ".json")

                if os.path.exists(json_path):
                    samples.append((img_path, json_path))

print(f"Total images found: {len(samples)}")

# ==============================
# SHUFFLE AND SPLIT
# ==============================

random.shuffle(samples)

total = len(samples)
train_end = int(total * 0.7)
val_end = int(total * 0.9)

splits = {
    "train": samples[:train_end],
    "val": samples[train_end:val_end],
    "test": samples[val_end:]
}

print("Train:", len(splits["train"]))
print("Val:", len(splits["val"]))
print("Test:", len(splits["test"]))

# ==============================
# CROP AND SAVE
# ==============================

def crop_and_save(split_name, data):
    for img_path, json_path in tqdm(data, desc=f"Processing {split_name}"):

        image = Image.open(img_path).convert("RGB")

        with open(json_path, "r") as f:
            annotation = json.load(f)

        for idx, shape in enumerate(annotation.get("shapes", [])):

            if shape.get("shape_type") != "rectangle":
                continue

            label = shape.get("label")

            if label in ["bridge_ahead", "right_hand_curve", "left_hand_curve"]:
                label = "road_sign"

            if label not in CLASSES:
                print(f"Label '{label}' not in CLASSES")
                continue

            points = shape["points"]
            (x1, y1), (x2, y2) = points

            x_min = int(min(x1, x2))
            y_min = int(min(y1, y2))
            x_max = int(max(x1, x2))
            y_max = int(max(y1, y2))

            crop = image.crop((x_min, y_min, x_max, y_max))

            base_name = Path(img_path).stem
            save_name = f"{base_name}_{idx}.jpg"

            save_path = os.path.join(
                OUTPUT_DIR,
                split_name,
                label,
                save_name
            )

            crop.save(save_path)


for split_name, data in splits.items():
    crop_and_save(split_name, data)

print("\n✅ Dataset creation complete.")
