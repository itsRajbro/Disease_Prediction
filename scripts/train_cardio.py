import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Ensure models directory exists
os.makedirs('model', exist_ok=True)

# Load both datasets
train_df = pd.read_csv(r'C:\Users\rajay\disease_predictor\disease_predictor\combined_dataset_cardio\train.csv')
val_df = pd.read_csv(r'C:\Users\rajay\disease_predictor\disease_predictor\combined_dataset_cardio\val.csv')

# Combine train and val datasets
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# Data cleaning - remove unrealistic blood pressure values
combined_df = combined_df[(combined_df['ap_hi'] >= 50) & (combined_df['ap_hi'] <= 250)]
combined_df = combined_df[(combined_df['ap_lo'] >= 40) & (combined_df['ap_lo'] <= 150)]

# Feature engineering - calculate BMI
combined_df['bmi'] = combined_df['weight'] / (combined_df['height']/100)**2

# Encode categorical variables
combined_df = pd.get_dummies(combined_df, columns=['cholesterol', 'gluc'], drop_first=True)

# Define features and target
X = combined_df.drop('cardio', axis=1)
y = combined_df['cardio']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model/cardio_model.joblib')
joblib.dump(scaler, 'model/cardio_scaler.joblib')

# Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("Model and scaler saved to models/cardio_model.joblib and models/cardio_scaler.joblib")
