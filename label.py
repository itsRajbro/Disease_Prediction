import pandas as pd

csv_path = "C:/Users/rajay/.cache/kagglehub/datasets/surajghuwalewala/ham1000-segmentation-and-classification/versions/2/GroundTruth.csv"
df = pd.read_csv(csv_path)

print("📄 Available columns in GroundTruth.csv:")
print(df.columns)
