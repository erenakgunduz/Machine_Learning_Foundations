import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

fh = logging.StreamHandler()
fmt = logging.Formatter(
    "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fmt)
logger.addHandler(fh)

# --- all tuning parameters ---
l = np.logspace(-2, 6, 9)  # lambda, from 1e-2 to 1e6, and with nine total samples
a = np.linspace(0, 1, 6)  # alpha, from 0 to 1, now with six evenly spaced samples


def preprocess_data(filename: str) -> tuple:
    "Take in raw data and convert it to a workable format/state"
    if not isinstance(filename, str):
        raise TypeError("Filename should be a string :)")

    try:
        datafile = f"{os.getcwd()}/../data/{filename}"
        logger.debug(datafile)
        if not os.path.exists(datafile):
            raise OSError("Expected data file, didn't find it :/")

        df = pd.read_csv(datafile, sep=",")  # read and pass to dataframe
        # debug shows that scikit-learn has mapped male to 1 and female to 0
        df[["Gender", "Student", "Married"]] = OrdinalEncoder().fit_transform(
            df[["Gender", "Student", "Married"]]
        )
        logger.debug(df[["Gender", "Student", "Married"]].head(10))

        # convert dataframe to numpy array for faster computations
        return (df.columns.to_numpy(), np.array(df))
    except OSError:
        print("Couldn't load in the data due to OS error.")
        sys.exit("Check if things are right and try again :)")


def standardize(data) -> tuple:
    "Establish design matrix and response vector, prepare both for elastic net"
    y = data[:, 9]  # extract only the data from the output column (balance)
    y = (lambda c: c - c.mean())(y)  # IIFE to center response vector
    logger.debug(y.shape)
    logger.debug(y.mean())

    dm = np.delete(data, 9, axis=1)  # extract the design matrix
    X = StandardScaler().fit_transform(dm)  # standardize (center & scale)

    logger.debug(X.shape)
    logger.debug([np.mean(X[:, k]) for k in range(X.shape[1])])
    logger.debug([np.std(X[:, k]) for k in range(X.shape[1])])
    return (X, y)


def elastic_net(X, y, l, a, cv: bool = False, k: int = 5) -> np.ndarray:
    if not isinstance(k, int):
        raise TypeError("Number of folds should be an integer :)")

    coeffs = np.zeros((6, 9, 9))
    cv_errors = np.zeros((9, 6, 5))

    def en(l, a):
        elastic = ElasticNet(alpha=l, l1_ratio=a)
        elastic.fit(X, y)
        return elastic.coef_

    if not isinstance(l, (int, float)) and not isinstance(a, (int, float)):
        for i_a, val_a in enumerate(np.flip(a)):
            for i_l, val_l in enumerate(l):
                if cv:
                    cv_errors[i_l, i_a] = np.abs(
                        cross_val_score(
                            ElasticNet(alpha=val_l, l1_ratio=val_a),
                            X,
                            y,
                            cv=k,
                            scoring="neg_mean_squared_error",
                        )
                    )
                else:
                    coeffs[i_a, i_l] = en(val_l, val_a)
    else:
        coeffs = en(l, a)

    if cv:
        logger.debug(f"{cv_errors.shape}\n{cv_errors}")
        logger.debug(cv_errors.mean(axis=2))
        return cv_errors.mean(axis=2)
    return coeffs


def main():
    columns, data = preprocess_data("Credit_N400_p9.csv")  # unpack the tuple
    # --- Deliverable 1 ---
    X, y = standardize(data)
    B = elastic_net(X, y, l, a)
    logger.debug(f"{B.shape}\n{B}")
    for index, alpha in enumerate(B):
        plt.figure(figsize=(8, 6))
        plt.xscale("log")
        # transpose so that each row is one of the nine features with the nine columns for TP
        # this way, each index (row) has the vector I need to plot points
        [plt.plot(l, b, label=f"{columns[i]}") for i, b in enumerate(alpha.T)]
        plt.xlabel(r"Tuning parameter ($\lambda$)")
        plt.ylabel(r"Regression coefficients ($\hat{\beta}$)")
        plt.legend(title="Features", fontsize="small")
        plt.savefig(f"../img/assign2/deliverable1_{index}_d6.png", dpi=200)
    # --- Deliverable 2 ---
    cv_error = elastic_net(X, y, l, a, True)
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    [plt.plot(l, cv, label=f"{round(a[i], 1)}") for i, cv in enumerate(cv_error.T)]
    plt.xlabel(r"Tuning parameter ($\lambda$)")
    plt.ylabel(r"$CV_{(5)}$ mean squared error")
    plt.legend(title=r"$\alpha$", fontsize="small")
    plt.savefig("../img/assign2/deliverable2_d6.png", dpi=200)
    # --- Deliverable 3 ---
    logger.debug(cv_error.argmin())
    logger.debug(cv_error.min())
    logger.debug(
        cv_error[
            cv_error.argmin() // cv_error.shape[1],  # gets the row
            cv_error.argmin() % cv_error.shape[1],  # gets the column
        ]
    )
    l_optimal = float(l[cv_error.argmin() // cv_error.shape[1]])
    a_optimal = float(a[cv_error.argmin() % cv_error.shape[1]])
    print(l_optimal, a[-(np.where(a == a_optimal)[0][0]) - 1])
    # --- Deliverable 4 ---
    B = elastic_net(X, y, l_optimal, a_optimal)
    print(B)
    B = elastic_net(X, y, l_optimal, a[5])  # lasso
    print(B)
    B = elastic_net(X, y, l_optimal, a[0])  # ridge
    print(B)


if __name__ == "__main__":
    main()
