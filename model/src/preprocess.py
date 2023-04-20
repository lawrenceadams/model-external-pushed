from sklearn.model_selection import train_test_split


# =================================================
"""
INTRODUCTION
This file serves as your data preprocessing script.

The function takes a Pandas DataFrame as its input, and MUST return two or four numpy arrays - one each for training inputs, training labels,
and the option to return testing inputs and testing labels.

The function name MUST be left as preprocess_data(), and should NOT be changed.
"""


def preprocess_data(df):
    """
    YOUR TRAINING SCRIPT GOES HERE
    Args:
        df (Pandas DataFrame): 
    Returns:
        X_train: a numpy array for training data e.g., shape (samples, features) or (samples, timesteps, features)
        y_train: a numpy array for training labels e.g., shape (samples) or (samples, 1)
        X_test (optional): a numpy array for testing data e.g., shape (samples, features) or (samples, timesteps, features)
        y_test (optional): a numpy array for testing labels e.g., shape (samples) or (samples, 1)
    """
    X_test = None
    y_test = None

    # Example script

    data = df.to_numpy()
    X, y = data[:, :5], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.25, random_state=42)

    """
    Your function MUST return, in the following order:
    1) X_train
    2) y_train
    3) X_test (or None)
    4) y_test (or None)
    """

    return X_train, y_train, X_test, y_test
