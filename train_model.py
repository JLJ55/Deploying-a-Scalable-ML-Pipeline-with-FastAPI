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
data = pd.read_csv(data_path)

# Split the data into train and test datasets
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
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)

# Save the model and the encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")

print(f"Saving the model to {model_path}")
save_model(model, model_path)

print(f"Saving the encoder to {encoder_path}")
save_model(encoder, encoder_path)

# Load the model to validate saving worked correctly
print("Loading the saved model...")
model = load_model(model_path)

# Run model inferences on the test dataset
print("Running inference on the test dataset...")
preds = inference(model, X_test)

# Calculate and print the metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute the performance on model slices
slice_output_path = os.path.join(project_path, "slice_output.txt")
print(f"Computing performance metrics on slices. Results will be saved to {slice_output_path}")

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
