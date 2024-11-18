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

# Split the provided data into train and test datasets
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
    X=train,  # Explicitly passing the train data as X
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    X=test,  # Explicitly passing the test data as X
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# Load the model to validate saving worked correctly
model = load_model(model_path)

# Run model inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute the performance on model slices
with open("slice_output.txt", "w") as f:
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
