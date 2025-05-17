import os
import shutil
import random
from PIL import Image

# Source folder (all images are tumor-positive)
tumor_src_folder = os.path.expanduser(
    r"C:\Users\rajay\.cache\kagglehub\datasets\pkdarabi\medical-image-dataset-brain-tumor-detection\versions\5\BrainTumor\BrainTumorYolov11\train"
)

# Target combined folder
combined_root = "combined_dataset_tumor"
img_size = (150, 150)

# Only one class: YES (tumor-positive)
disease_class = "YES"

def prepare_dataset():
    # Create folders for YES class only
    for dtype in ["train", "val"]:
        dest = os.path.join(combined_root, dtype, disease_class)
        os.makedirs(dest, exist_ok=True)

    # List and shuffle images
    images = [f for f in os.listdir(tumor_src_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img_list, dtype in [(train_imgs, "train"), (val_imgs, "val")]:
        for img_name in img_list:
            src_path = os.path.join(tumor_src_folder, img_name)
            dest_path = os.path.join(combined_root, dtype, disease_class, img_name)

            try:
                with Image.open(src_path) as img:
                    img = img.resize(img_size)
                    img.save(dest_path)
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")

    print("✅ Tumor dataset has been organized successfully.")

if __name__ == "__main__":
    prepare_dataset()
