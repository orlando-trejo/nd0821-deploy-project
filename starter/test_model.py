import pytest
import numpy as np
from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    model = train_model(X_train, y_train, 2)
    assert model is not None
    assert hasattr(model, "predict")

def test_compute_model_metrics():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == pytest.approx(0.666666, 0.01)

def test_inference():
    X = np.array([[10, 20], [30, 40], [50, 60]])
    y = np.array([0, 1, 0])
    model = train_model(X, y, 2)
    preds = inference(model, X)
    assert len(preds) == len(y)
