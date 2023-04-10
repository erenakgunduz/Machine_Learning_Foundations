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

# --- all tuning parameters ---
lambdas = np.logspace(-2, 6, 9)  # from 1e-2 to 1e6, and with nine total samples
alphas = np.linspace(0, 1, 6)  # from 0 to 1, now with six evenly spaced samples


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


def elastic_net(data) -> tuple:
    "Establish design matrix and response vector, prepare both for elastic net"
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


def coordinate_descent(X, y, lmbd, a) -> np.ndarray:
    "Implementation of vectorized coordinate descent, applying elastic net"

    def cd(lmbd, a):
        # b vector, each value (b_k) remains constant
        b = np.array([np.sum(X[:, k] ** 2) for k in range(X.shape[1])])
        # starting parameters vector
        beta = np.array([np.random.rand() for _ in range(X.shape[1])])
        for _ in range(1000):  # total iterations
            for k, _ in enumerate(beta):
                x_k = X[:, k]
                a_k = x_k.T @ (y - X @ beta + x_k * beta[k])
                relu = np.maximum(0, np.abs(a_k) - (lmbd * (1 - a) / 2))
                beta[k] = np.sign(a_k) * relu / (b[k] + lmbd * a)
        return beta

    coeffs = np.zeros((6, 9, 9))

    if not isinstance(lmbd, (int, float)) and not isinstance(a, (int, float)):
        for i_a, val_a in enumerate(a):
            for i_l, val_l in enumerate(lmbd):
                beta = cd(val_l, val_a)
                coeffs[i_a, i_l] = beta
    else:
        coeffs = cd(lmbd, a)
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
        train_X, train_y = elastic_net(train)
        val_X, val_y = elastic_net(validation)
        b = coordinate_descent(train_X, train_y, lambdas, alphas)
        # after preparing and training data, once again check that things look ok
        logger.debug(
            f"{lambdas.shape} {b.shape} {val_X.shape} {b[0].shape} {val_y.shape[0]}"
        )
        mse = [
            [
                (
                    (val_y - val_X @ b[i_a, i_l]).T
                    @ (val_y - val_X @ b[i_a, i_l])
                    / val_y.shape[0]
                )
                for i_l, _ in enumerate(lambdas)
            ]
            for i_a, _ in enumerate(b)
        ]
        cv_errors.append(mse)

    cv_error = np.array(cv_errors).T
    logger.debug(f"{cv_error.shape} {cv_error}")
    cv_error = cv_error.mean(axis=2)
    logger.debug(f"{cv_error.shape} {cv_error}")
    return cv_error


def main():
    columns, data = preprocess_data("Credit_N400_p9.csv")  # unpack the tuple
    # --- Deliverable 1 ---
    X, y = elastic_net(data)
    B = coordinate_descent(X, y, lambdas, alphas)
    logger.debug(f"{B.shape}\n{B}")
    for index, alpha in enumerate(B):
        plt.figure(figsize=(8, 6))
        plt.xscale("log")
        # transpose so that each row one of the nine features with nine columns for TP
        # this way, each index (row) has the vector I need to plot points
        [plt.plot(lambdas, b, label=f"{columns[i]}") for i, b in enumerate(alpha.T)]
        plt.xlabel(r"Tuning parameter ($\lambda$)")
        plt.ylabel(r"Regression coefficients ($\hat{\beta}$)")
        plt.legend(title="Features", fontsize="small")
        plt.savefig(f"{file_path}/../img/assign2/deliverable1_{index}.png", dpi=200)
    # --- Deliverable 2 ---
    cv_error = cross_validation(data)
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    [
        plt.plot(lambdas, cv, label=f"{round(alphas[i], 1)}")
        for i, cv in enumerate(cv_error.T)
    ]
    plt.xlabel(r"Tuning parameter ($\lambda$)")
    plt.ylabel(r"$CV_{(5)}$ mean squared error")
    plt.legend(title=r"$\alpha$", fontsize="small")
    plt.savefig(f"{file_path}/../img/assign2/deliverable2.png", dpi=200)
    # --- Deliverable 3 ---
    logger.debug(cv_error.argmin())
    logger.debug(cv_error.min())
    logger.debug(
        cv_error[
            cv_error.argmin() // cv_error.shape[1],  # gets the row
            cv_error.argmin() % cv_error.shape[1],  # gets the column
        ]
    )
    l_optimal = float(lambdas[cv_error.argmin() // cv_error.shape[1]])
    a_optimal = float(alphas[cv_error.argmin() % cv_error.shape[1]])
    print(l_optimal, a_optimal)
    # --- Deliverable 4 ---
    B = coordinate_descent(X, y, l_optimal, a_optimal)
    print(B)
    B = coordinate_descent(X, y, l_optimal, alphas[0])  # lasso
    print(B)
    B = coordinate_descent(X, y, l_optimal, alphas[5])  # ridge
    print(B)


if __name__ == "__main__":
    main()
