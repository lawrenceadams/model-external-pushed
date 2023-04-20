from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# =================================================
"""
INTRODUCTION
This file serves as your model training script for AML.

The function takes numpy arrays as its input - these can be training and testing arrays,
or just the training arrays.

The function name MUST be left as train_model, and should NOT be changed.
"""


def train_model(X_train, y_train, X_test=None, y_test=None):
    """
    YOUR TRAINING SCRIPT GOES HERE

    Args:
        X_train (numpy array): training data as a numpy array
        y_train (numpy array): training labels as a numpy array
        X_test (numpy array) [OPTIONAL]: testing data as a numpy array
        y_test (numpy array) [OPTIONAL]: testing data as a numpy array
    Returns:
        model: fitted model
        train_metrics: training metrics
        test_metrics: testing metrics
    """

    # Example script

    # Define the model
    model = XGBClassifier()

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_train_pred = model.predict(X_train)
    if X_test != None:
        y_test_pred = model.predict(X_test)

    train_metrics = {'train_accuracy': accuracy_score(y_train, y_train_pred),
                     'train_roc_auc_score': roc_auc_score(y_train, y_train_pred)}

    if X_test != None:
        test_metrics = {'test_accuracy': accuracy_score(y_test, y_test_pred),
                        'test_roc_auc_score': roc_auc_score(y_test, y_test_pred)}
    else:
        test_metrics = {}

    """
    Your function MUST return, in the following order:
    1) A working model
    2) A dictionary of your desired training metrics
    3) A dictionary of your desired testing metrics
    """

    return model, train_metrics, test_metrics
