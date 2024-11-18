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

# Set project and data paths
project_path = "/home/acj71/Udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")

# Load the census.csv data
print(f"Loading data from {data_path}")
data = pd.read_csv(data_path)

# Split the data into a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Process the data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
print(f"Saving model to {model_path}")
save_model(model, model_path)
print(f"Saving encoder to {encoder_path}")
save_model(encoder, encoder_path)

# Load the model
print(f"Loading model from {model_path}")
model = load_model(model_path)

# Run model inferences on the test dataset
print("Running inferences on the test dataset...")
preds = inference(model, X_test)

# Calculate and print the metrics
print("Calculating performance metrics...")
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices
print("Computing performance on model slices...")
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test, col, slicevalue, cat_features, "salary", encoder, lb, model
            )
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
print(f"Slice metrics saved to {slice_output_path}")
