import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = "YOUR_PROJECT_PATH_HERE"  # Replace with your actual project path
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split the data into a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the train data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model on the training dataset
print("Training the model...")
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(model, model_path)
save_model(encoder, encoder_path)
print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")

# Load the model
print("Loading the saved model...")
model = load_model(model_path)

# Run inference on the test dataset
print("Running inference on test data...")
preds = inference(model, X_test)

# Calculate and print overall metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall Metrics - Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute performance metrics for data slices
print("Computing performance metrics for data slices...")
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            precision, recall, fbeta = performance_on_categorical_slice(
                test,
                column_name=col,
                slice_value=slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            f.write(f"{col}: {slice_value}, Count: {count:,}\n")
            f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n\n")
            print(f"{col}: {slice_value}, Count: {count:,}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

print(f"Performance metrics for data slices saved to {slice_output_path}")
