# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, sliced_metrics
import pandas as pd
import numpy as np
import logging
import pickle
import json


# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="train_model.log",
)

# Add code to load in the data.
logging.info("Loading data")
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.replace(" ", "")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)
logging.info(f"Training data shape: {train.shape}, Testing data shape: {test.shape}")

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
logging.info("Processing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
logging.info("Processing testing data")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
logging.info("Training the model")
model = train_model(X_train, y_train)
logging.info("saving model and artifacts")
with open ("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

# Evaluate the model
logging.info("Evaluating the model")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Save metrics
metrics = {
    "precision": float(precision),
    "recall": float(recall),
    "fbeta": float(fbeta),
}
logging.info(f"Metrics: {metrics}")
with open("../model/metrics.json", "w") as f:
    json.dump(metrics, f)


# Calculate sliced metrics
logging.info("Calculating sliced metrics")
all_slice_metrics = sliced_metrics(model, X_test, y_test, test, cat_features)
with open("../model/slice_metrics.json", "w") as f:
    json.dump(all_slice_metrics, f)
with open("../model/slice_output.txt", "w") as f:
    for k, v in all_slice_metrics.items():
        f.write(f"{k}: {v}\n")
logging.info("Slice metrics saved")