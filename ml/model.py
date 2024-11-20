import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or encoder.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads pickle file from `path` and returns it.

    Returns
    -------
    model
        Loaded machine learning model or encoder.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(data, column_name, slice_value, categorical_features, label, encoder, lb, model):
    """
    Computes the model metrics on a slice of the data specified by a column name.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, or float
        Value of the slice feature.
    categorical_features: list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : sklearn model
        Trained model used for inference.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Filter data for the slice
    sliced_data = data[data[column_name] == slice_value]

    # Process the sliced data
    X_slice, y_slice, _, _ = process_data(
        X=sliced_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make predictions
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
