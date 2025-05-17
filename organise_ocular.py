import os
import shutil
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
base_path = r"C:\Users\rajay\.cache\kagglehub\datasets\andrewmvd\ocular-disease-recognition-odir5k\versions\2\ODIR-5K"
csv_path = os.path.join(base_path, "data.xlsx") 


train_img_dir = os.path.join(base_path, "Training Images")
test_img_dir = os.path.join(base_path, "Testing Images")
output_dir = "combined_dataset_ocular"
img_size = (224, 224)

# Disease labels

labels = ["N", "D", "G", "C", "A", "H", "M", "O"]

# Create folders for each split and class
splits = ["train", "val", "test"]
for split in splits:
    for label in labels:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

def copy_and_resize(src_path, dest_dirs):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(img_size)
            for dest_dir in dest_dirs:
                dest_path = os.path.join(dest_dir, os.path.basename(src_path))
                img.save(dest_path)
    except Exception as e:
        print(f"⚠️ Error processing {src_path}: {e}")

def prepare_ocular_dataset():
    df = pd.read_excel(csv_path)

    # Fix any leading/trailing whitespaces in column names
    df.columns = df.columns.str.strip()
    print("Available columns:", df.columns.tolist())

    # Drop rows with missing image paths
    df = df.dropna(subset=["Left-Fundus", "Right-Fundus"])

    # Split the data by patient ID
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    def process_split(data, split):
        for _, row in data.iterrows():
            for side in ["Left", "Right"]:
                img_name = row[f"{side}-Fundus"]
                label_flags = [row[label] for label in labels if label in row]

                # Handle case where no label = skip
                if sum(label_flags) == 0:
                    continue

                # Find actual path (train or test folder)
                src_path = os.path.join(train_img_dir, img_name)
                if not os.path.exists(src_path):
                    src_path = os.path.join(test_img_dir, img_name)
                if not os.path.exists(src_path):
                    print(f"⚠️ Image not found: {img_name}")
                    continue

                # Get output folders for each label
                target_dirs = [os.path.join(output_dir, split, label) for i, label in enumerate(labels) if label_flags[i] == 1]
                copy_and_resize(src_path, target_dirs)

    print("🚧 Processing training data...")
    process_split(train_df, "train")
    print("🚧 Processing validation data...")
    process_split(val_df, "val")
    print("🚧 Processing test data...")
    process_split(test_df, "test")

    print("✅ Ocular dataset organized successfully.")

if __name__ == "__main__":
    prepare_ocular_dataset()
