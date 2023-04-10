import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

fh = logging.StreamHandler()
fmt = logging.Formatter(
    "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fmt)
logger.addHandler(fh)

file_path = os.path.dirname(os.path.realpath(__file__))


# grid of tuning parameters
lambdas = np.logspace(-2, 4, 7)


def preprocess_data(filename: str) -> tuple:
    "Take in raw data and convert it to a workable format/state"
    if not isinstance(filename, str):
        raise TypeError("Filename should be a string :)")

    try:
        datafile = f"{file_path}/../data/{filename}"
        logger.debug(datafile)
        if not os.path.exists(datafile):
            raise OSError("Expected data file, didn't find it :/")

        df = pd.read_csv(datafile, sep=",")  # read and pass to dataframe
        mapping = {
            "Gender": {"Female": 0, "Male": 1},
            "Student": {"No": 0, "Yes": 1},
            "Married": {"No": 0, "Yes": 1},
        }  # so no ambiguity about how we encode categorical columns
        df = df.replace(mapping)

        # convert dataframe to numpy array for faster computations
        return (df.columns.to_numpy(), np.array(df))
    except OSError:
        print("Couldn't load in the data due to OS error.")
        sys.exit("Check if things are right and try again :)")


def ridge_regression(data) -> tuple:
    "Establish design matrix and response vector, prepare both for ridge regression"
    y = data[:, 9]  # extract only the data from the output column (balance)
    y = (lambda c: c - c.mean())(y)  # IIFE to center response vector
    logger.debug(y.shape)
    logger.debug(y.mean())

    dm = np.delete(data, 9, axis=1)  # extract the design matrix
    X = (dm - np.mean(dm, axis=0)) / np.std(dm, axis=0)  # standardize (center & scale)

    logger.debug(X.shape)
    logger.debug([np.mean(X[:, k]) for k in range(X.shape[1])])
    logger.debug([np.std(X[:, k]) for k in range(X.shape[1])])
    return (X, y)


def gradient_descent(X, y, lmbd, a=10**-5) -> np.ndarray:
    "Vectorized batch gradient descent w/ learning rate alpha applying ridge regression"

    def gd(lmbd):
        # starting parameters vector
        b = np.array([np.random.rand() for _ in range(X.shape[1])])
        for _ in range(10**5):  # total iterations for each tuning parameter
            b = b - 2 * a * (lmbd * b - X.T @ (y - X @ b))
        return b

    coeffs = np.zeros((7, 9))
    logger.debug(coeffs)

    if not isinstance(lmbd, (int, float)):
        for index, val in enumerate(lmbd):
            b = gd(val)
            coeffs[index] = b
    else:
        coeffs = gd(lmbd)
    return coeffs


def cross_validation(data, k: int = 5) -> np.ndarray:
    "Implementation of relevant gradient descent utilizing k-fold cross validation"
    if not isinstance(k, int):
        raise TypeError("Number of folds should be an integer :)")

    # np.random.seed(0)
    data_shuffled = np.random.permutation(data)  # shuffled copy of the original data
    logger.debug(f"{data_shuffled.shape}\n{data_shuffled}")

    # create tensor containing each equally shaped fold
    folds = np.array(np.split(data_shuffled, k))  # splits row-wise by default
    logger.debug(f"{folds.shape}\n{folds}")  # confirm that things look right

    cv_errors = []
    for index, fold in enumerate(folds):
        logger.debug("\n")
        logger.debug(index)

        train = np.delete(folds, index, axis=0)
        train = train.reshape(-1, train.shape[2])
        validation = fold
        # logger.debug(f"{train.shape}\n{validation.shape}")
        # logger.debug(train[:1])
        # logger.debug(validation[:1])
        train_X, train_y = ridge_regression(train)
        val_X, val_y = ridge_regression(validation)
        b = gradient_descent(train_X, train_y, lambdas)
        # after preparing and training data, once again check that things look ok
        logger.debug(
            f"{lambdas.shape} {b.shape} {val_X.shape} {b[0].shape} {val_y.shape[0]}"
        )
        mse = [
            ((val_y - val_X @ b[i]).T @ (val_y - val_X @ b[i]) / val_y.shape[0])
            for i, _ in enumerate(lambdas)
        ]
        cv_errors.append(mse)

    cv_error = np.array(cv_errors).T
    logger.debug(f"{cv_error.shape} {cv_error}")
    cv_error = np.array([lmbd.mean() for lmbd in cv_error])
    logger.debug(f"{cv_error.shape} {cv_error}")
    return cv_error


def main():
    columns, data = preprocess_data("Credit_N400_p9.csv")  # unpack the tuple
    # --- Deliverable 1 ---
    X, y = ridge_regression(data)
    # transpose so that each row one of the nine features with the seven columns for TP
    b = gradient_descent(X, y, lambdas).T  # use default value for learning rate
    # this way, each index (row) has the vector I need to plot points
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    [plt.plot(lambdas, b, label=f"{columns[i]}") for i, b in enumerate(b)]
    plt.xlabel(r"Tuning parameter ($\lambda$)")
    plt.ylabel(r"Regression coefficients ($\hat{\beta}$)")
    plt.legend(title="Features", fontsize="small")
    plt.savefig(f"{file_path}/../img/assign1/deliverable1.png", dpi=200)
    # --- Deliverable 2 ---
    cv_error = cross_validation(data)
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.plot(lambdas, cv_error)
    plt.xlabel(r"Tuning parameter ($\lambda$)")
    plt.ylabel(r"$CV_{(5)}$ mean squared error")
    plt.savefig(f"{file_path}/../img/assign1/deliverable2.png", dpi=200)
    # --- Deliverable 3 ---
    l_optimal = float(lambdas[cv_error.argmin()])
    print(l_optimal)
    # --- Deliverable 4 ---
    b = gradient_descent(X, y, l_optimal)
    print(b)


if __name__ == "__main__":
    main()
