import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Correct path to the CSV file
csv_path = r"C:\Users\rajay\.cache\kagglehub\datasets\sulianova\cardiovascular-disease-dataset\versions\1\cardio_train.csv"  # or use the exact path on your system

# Output directory
output_folder = "combined_dataset_cardio"
os.makedirs(output_folder, exist_ok=True)

def prepare_cardio_dataset():
    # Load the CSV
    df = pd.read_csv(csv_path, sep =';')
    print("Available columns:", df.columns.tolist())


    # Optional: check for missing values
    if df.isnull().sum().sum() > 0:
        print("⚠️ Dataset contains missing values. Filling with median.")
        df.fillna(df.median(), inplace=True)

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cardio'])

    # Save to CSV
    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_folder, "val.csv"), index=False)

    print("✅ Cardiovascular dataset split into train/val CSV files.")

if __name__ == "__main__":
    prepare_cardio_dataset()
