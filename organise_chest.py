import os
from PIL import Image
import shutil
import random

# Paths
source_dir = r"C:\Users\rajay\.cache\kagglehub\datasets\jtiptj\chest-xray-pneumoniacovid19tuberculosis\versions\1"
target_dir = "combined_dataset_cxr"
img_size = (224, 224)


disease_classes = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
splits = ["train", "val", "test"]

def prepare_dataset():
    for split in splits:
        for cls in disease_classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

            src_folder = os.path.join(source_dir, split, cls)
            dest_folder = os.path.join(target_dir, split, cls)

            for img_name in os.listdir(src_folder):
                src_path = os.path.join(src_folder, img_name)
                dest_path = os.path.join(dest_folder, img_name)

                try:
                    with Image.open(src_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(img_size)
                        img.save(dest_path)
                except Exception as e:
                    print(f"⚠️ Error processing {img_name}: {e}")

    print("✅ Chest X-ray dataset organized and resized successfully.")

if __name__ == "__main__":
    prepare_dataset()
