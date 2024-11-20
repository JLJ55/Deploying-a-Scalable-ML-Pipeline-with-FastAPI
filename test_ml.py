import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

@pytest.fixture
def mock_data():
    data = {
        "age": [25, 32, 40, 28],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "State-gov"],
        "education": ["Bachelors", "HS-grad", "HS-grad", "Bachelors"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married"],
        "occupation": ["Tech-support", "Exec-managerial", "Adm-clerical", "Protective-serv"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Unmarried"],
        "race": ["White", "White", "Black", "White"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def processed_data(mock_data):
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
    X, y, encoder, lb = process_data(
        mock_data, categorical_features=cat_features, label="salary", training=True
    )
    return X, y, encoder, lb

# Test 1: Check if the model training function returns a valid model
def test_train_model(processed_data):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

# Test 2: Check if compute_model_metrics returns correct metrics
def test_compute_model_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert precision == pytest.approx(1.0, rel=1e-2)
    assert recall == pytest.approx(1.0, rel=1e-2)
    assert f1 == pytest.approx(1.0, rel=1e-2)

# Test 3: Check if the inference function produces predictions of the right shape
def test_inference(processed_data):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape
