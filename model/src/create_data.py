import os
import argparse
import pandas as pd
import numpy as np
from preprocess import preprocess_data
import logging
import mlflow


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    df = pd.read_csv(args.data, header=0)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", df.shape[1] - 1)

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(df)
    
    # Save the arrays
    for path, names, arrs in [[args.train_data, ['train_data_X', 'train_data_y'], [X_train, y_train]],
                              [args.test_data, ['test_data_X', 'test_data_y'], [X_test, y_test]]]:
        for name, arr in zip(names, arrs):
            np.save(os.path.join(path, name), arr)
    
    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
