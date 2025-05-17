import os
import shutil
import random
from PIL import Image

# Path to your extracted malaria dataset
malaria_path = os.path.expanduser(r"C:\Users\rajay\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1\cell_images\cell_images")  # CHANGE this to your actual path

# Output folder
combined_root = "combined_dataset_malaria"
img_size = (150, 150)

disease_classes = ["Parasitized", "Uninfected"]

def prepare_dataset():
    for dtype in ["train", "val"]:
        for cls in disease_classes:
            dest = os.path.join(combined_root, dtype, cls)
            os.makedirs(dest, exist_ok=True)

    for cls in disease_classes:
        src_folder = os.path.join(malaria_path, cls)
        images = os.listdir(src_folder)
        random.shuffle(images)

        split_idx = int(0.8 * len(images))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img_list, dtype in [(train_imgs, "train"), (val_imgs, "val")]:
            for img_name in img_list:
                src_path = os.path.join(src_folder, img_name)
                dest_path = os.path.join(combined_root, dtype, cls, img_name)

                try:
                    with Image.open(src_path) as img:
                        img = img.resize(img_size)
                        img.save(dest_path)
                except Exception as e:
                    print(f"⚠️ Error processing {img_name}: {e}")

    print("✅ Malaria dataset has been organized successfully.")

if __name__ == "__main__":
    prepare_dataset()
