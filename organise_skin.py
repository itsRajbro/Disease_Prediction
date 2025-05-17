import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
base_path = r"C:\Users\rajay\.cache\kagglehub\datasets\surajghuwalewala\ham1000-segmentation-and-classification\versions\2"
csv_path = os.path.join(base_path, "GroundTruth.csv")
image_dir = os.path.join(base_path, "images")  # Adjust folder name if needed
mask_dir = os.path.join(base_path, "masks")    # Adjust folder name if needed
output_dir = "combined_dataset_skin"
img_size = (224, 224)

# Create output folders
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Optional: If 'image' column doesn't include ".jpg" or full filename, add extension
if not df['image'].iloc[0].endswith(".jpg"):
    df["image"] = df["image"].astype(str) + ".jpg"

# Split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

def process_split(data, split):
    for _, row in data.iterrows():
        img_name = row["image"]
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)  # Mask has same name as image

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"⚠️ Skipping missing file: {img_name}")
            continue

        try:
            # Process image
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(img_size)
                img.save(os.path.join(output_dir, split, "images", img_name))

            # Process mask
            with Image.open(mask_path) as mask:
                mask = mask.convert("L")  # grayscale
                mask = mask.resize(img_size)
                mask.save(os.path.join(output_dir, split, "masks", img_name))
        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

# Process all splits
print("🚧 Processing training data...")
process_split(train_df, "train")
print("🚧 Processing validation data...")
process_split(val_df, "val")
print("🚧 Processing test data...")
process_split(test_df, "test")

print("✅ Skin dataset organized successfully.")
