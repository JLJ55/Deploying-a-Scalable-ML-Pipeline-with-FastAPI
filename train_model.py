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

# Define the project path
project_path = "/home/acj71/Udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"

# Load the census.csv data
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at: {data_path}")

# Read the data
data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Split the provided data into train and test datasets
print("Splitting the dataset...")
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
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

# Process the training data
print("Processing training data...")
try:
    X_train, y_train, encoder, lb = process_data(
        X=train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
except Exception as e:
    print(f"Error processing training data: {e}")
    raise

# Process the test data
print("Processing test data...")
try:
    X_test, y_test, _, _ = process_data(
        X=test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
except Exception as e:
    print(f"Error processing test data: {e}")
    raise

# Train the model
print("Training the model...")
try:
    model = train_model(X_train, y_train)
    print("Model training completed.")
except Exception as e:
    print(f"Error training the model: {e}")
    raise

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")

try:
    print(f"Saving model to: {model_path}")
    save_model(model, model_path)
    print(f"Saving encoder to: {encoder_path}")
    save_model(encoder, encoder_path)
except Exception as e:
    print(f"Error saving model or encoder: {e}")
    raise

# Load the model to validate saving worked correctly
print("Loading the saved model...")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Run model inferences on the test dataset
print("Running inference on test data...")
try:
    preds = inference(model, X_test)
except Exception as e:
    print(f"Error during inference: {e}")
    raise

# Calculate and print the metrics
print("Calculating metrics...")
try:
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")
except Exception as e:
    print(f"Error calculating metrics: {e}")
    raise

# Compute the performance on model slices
slice_output_path = os.path.join(project_path, "slice_output.txt")
print("Computing performance on slices...")
try:
    with open(slice_output_path, "w") as f:
        for col in cat_features:
            for slice_value in sorted(test[col].unique()):
                count = test[test[col] == slice_value].shape[0]
                p, r, fb = performance_on_categorical_slice(
                    data=test,
                    column_name=col,
                    slice_value=slice_value,
                    categorical_features=cat_features,
                    label="salary",
                    encoder=encoder,
                    lb=lb,
                    model=model
                )
                f.write(f"{col}: {slice_value}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
    print(f"Slice output saved to: {slice_output_path}")
except Exception as e:
    print(f"Error computing slice performance: {e}")
    raise
