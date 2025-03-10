from sklearn.metrics import fbeta_score, precision_score, recall_score
#Import classificaiton models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, cv=5):
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
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }


    # Initialize the model
    model = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
    )
    # Fit the model
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
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


def sliced_metrics(model, X_test, y_test, test_df, cat_features):
    """
    Compute model metrics for multiple features.
    
    Inputs
    ------
    model : sklearn model
        Trained machine learning model
    X_test : np.array
        Processed test data
    y_test : np.array
        Test labels
    test_df : pd.DataFrame
        Original test dataframe with categorical features
    cat_features : list
        List of categorical feature names
        
    Returns
    -------
    all_slice_metrics : dict
        Dictionary of slice metrics
    """
    all_slice_metrics = {}
    
    for feature in cat_features:
        feature_metrics = {}
        for val in test_df[feature].unique():
            # Create mask on original dataframe
            mask = test_df[feature] == val
            # Apply mask to processed data (NumPy arrays)
            X_slice = X_test[mask.values]
            y_slice = y_test[mask.values]
            
            # Get predictions and metrics
            slice_preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, slice_preds)
            
            # Store metrics
            feature_metrics[str(val)] = {
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": float(fbeta),
                "samples": int(mask.sum())
            }
            
        all_slice_metrics[feature] = feature_metrics

    return all_slice_metrics
        


