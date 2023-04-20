import argparse
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import os
import numpy as np
import mlflow
from model import train_model


# =================================================
"""
Logging the model training process
"""

# Start Logging
mlflow.start_run()

# enable autologging - this is specific to scikit-learn models only
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


# =================================================
"""
Running the training script in AML

The main() function will load your training / testing data, and then run
your model-training script. 
"""


def main():

    def load_files(path):
        """
        This function provides the path to the training or testing files.
        It assumes the only files in the directory are the X and y arrays, in that order.

        Args:
            path (str): path to the parent directory
        Returns:
            X, y: two numpy arrays
        """
        files = os.listdir(path)
        
        arrays = {}
        for file in files:
            arr = np.load(os.path.join(path, file))
            if 'X' in file:
                arrays['X'] = arr
            elif 'y' in file.replace('.npy', ''):
                arrays['y'] = arr
            else:
                raise KeyError(f"Failure to load either the X or y arrays - Encountered file {file}.")
        
        return arrays['X'], arrays['y']

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # Locate the training/testing data
    X_train, y_train = load_files(args.train_data)
    X_test, y_test = load_files(args.test_data)

    print('Beginning training...')
    model, train_metrics, test_metrics = train_model(X_train, y_train, X_test, y_test)

    # Log the output from the training process
    mlflow.log_metrics(train_metrics)
    mlflow.log_metrics(test_metrics)

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
