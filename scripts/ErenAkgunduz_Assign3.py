import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# mypy: disable-error-code="call-overload"

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
lambdas = np.logspace(-4, 4, 9)


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

        # convert dataframe to numpy array for faster computations
        return (df.columns.to_numpy(), np.array(df))
    except OSError:
        print("Couldn't load in the data due to OS error.")
        sys.exit("Check if things are right and try again :)")


class Begin:
    "Establish design and response matrices"

    def __init__(self, data) -> None:
        self.data: np.ndarray = data
        self.classes: np.ndarray = np.copy(self.data[:, 10])
        self.labels: tuple = tuple(sorted(set(self.classes)))

    def initialize(self) -> tuple:
        # extract the design matrix
        dm = np.array(np.delete(self.data, 10, axis=1), dtype=float)
        # standardize (center & scale)
        dm = (dm - np.mean(dm, axis=0)) / np.std(dm, axis=0)
        aug = np.ones((dm.shape[0], 1))

        X = np.hstack((aug, dm))  # augmented design matrix

        for i, c in enumerate(self.classes):  # encode labels with binarize/one-hot
            if c == self.labels[0]:
                self.classes[i] = np.array([1, 0, 0, 0, 0])
            elif c == self.labels[1]:
                self.classes[i] = np.array([0, 1, 0, 0, 0])
            elif c == self.labels[2]:
                self.classes[i] = np.array([0, 0, 1, 0, 0])
            elif c == self.labels[3]:
                self.classes[i] = np.array([0, 0, 0, 1, 0])
            elif c == self.labels[4]:
                self.classes[i] = np.array([0, 0, 0, 0, 1])
            else:
                self.classes[i] = np.array([0, 0, 0, 0, 0])

        Y = np.vstack(self.classes)  # response matrix

        logger.debug(X.shape)
        logger.debug(X.mean(axis=0))
        logger.debug(X.std(axis=0))
        logger.debug(Y.shape)
        return (X, Y)


def gradient_descent(X, Y, lmbd, a=1e-5) -> np.ndarray:
    "Vectorized batch gradient descent, learning rate alpha, logistic ridge regression"

    def gd(lmbd):
        # starting parameter matrix
        B = np.zeros((X.shape[1], Y.shape[1]))
        for _ in range(10**4):  # total iterations for each tuning parameter
            U = np.exp(X @ B)  # unnormalized class probability matrix
            P = U / U.sum(axis=1, keepdims=True)  # normalized class probability matrix
            Z = np.vstack((B[0], np.zeros((X.shape[1] - 1, Y.shape[1]))))  # intercepts
            B = B + a * (X.T @ (Y - P) - 2 * lmbd * (B - Z))
        return B

    params = np.zeros((9, 11, 5))

    if not isinstance(lmbd, (int, float)):
        for index, val in enumerate(lmbd):
            b = gd(val)
            params[index] = b
    else:
        params = gd(lmbd)
    return params


def cross_validation(data, k: int = 5) -> np.ndarray:
    "Implementation of relevant gradient descent utilizing k-fold cross validation"
    if not isinstance(k, int):
        raise TypeError("Number of folds should be an integer :)")

    # np.random.seed(0)
    data_shuffled = np.random.permutation(data)  # shuffled copy of the original data
    logger.debug(f"{data_shuffled.shape}\n{data_shuffled}")

    # splits row-wise by default, will contain folds of varying shapes
    folds = np.array(np.array_split(data_shuffled, k), dtype=object)

    cv_errors = []
    for index, fold in enumerate(folds):
        logger.debug("\n")
        logger.debug(index)

        train = np.delete(folds, index, axis=0)
        train = np.vstack(train)
        validation = fold
        # logger.debug(f"{train.shape}\n{validation.shape}")
        # logger.debug(train[:1])
        # logger.debug(validation[:1])
        train_X, train_Y = Begin(train).initialize()
        val_X, val_Y = Begin(validation).initialize()
        B = gradient_descent(train_X, train_Y, lambdas)
        # use training parameters to test predictors
        U = np.array([np.exp(val_X @ b) for b in B])
        # normalize the new probability matrices
        P = np.array([u / u.sum(axis=1, keepdims=True) for u in U])
        # now that data is trained and prepared, once again check that things look ok
        logger.debug(f"{lambdas.shape} {P.shape} {val_Y.shape}")
        # obtain categorical cross-entropy loss by multiplying all values element-wise
        cce = np.array([np.sum(val_Y * np.log10(p)) * -(1 / val_Y.shape[0]) for p in P])
        logger.debug(f"{cce.shape}\n{cce}")
        cv_errors.append(cce)

    logger.debug(f"{np.array(cv_errors).T.shape}\n{np.array(cv_errors).T}")
    logger.debug(np.array(cv_errors).T.mean(axis=1))
    return np.array(cv_errors).T.mean(axis=1)


def main():
    columns, data = preprocess_data("TrainingData_N183_p10.csv")  # unpack the tuple
    # --- Deliverable 1 ---
    X, Y = Begin(data).initialize()
    # transpose so that each row one of the five classes
    B = gradient_descent(X, Y, lambdas).T
    # logger.debug(f"{B.shape}\n{B}")

    B_ni = np.zeros((5, 10, 9))  # parameters only, no intercepts
    for i, k in enumerate(B):
        B_ni[i] = np.delete(k, 0, axis=0)
    # logger.debug(f"{B_ni.shape}\n{B_ni}")

    for index, cat in enumerate(B_ni):
        plt.figure(figsize=(8, 6))
        plt.xscale("log")
        [plt.plot(lambdas, b, label=f"{columns[i]}") for i, b in enumerate(cat)]
        plt.xlabel(r"Tuning parameter ($\lambda$)")
        plt.ylabel(r"Regression coefficients ($\hat{\beta}$)")
        plt.legend(title="Features", fontsize="small")
        plt.savefig(
            f"{file_path}/../img/assign3/deliverable1_{Begin(data).labels[index]}.png",
            dpi=200,
        )
    # --- Deliverable 2 ---
    cv_errors = cross_validation(data)
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.plot(lambdas, cv_errors)
    plt.xlabel(r"Tuning parameter ($\lambda$)")
    plt.ylabel(r"$CV_{(5)}$ categorical cross-entropy loss")
    plt.savefig(f"{file_path}/../img/assign3/deliverable2.png", dpi=200)
    # --- Deliverable 3 ---
    l_optimal = float(lambdas[cv_errors.argmin()])
    print(l_optimal)
    # --- Deliverable 4 ---
    B = gradient_descent(X, Y, l_optimal)  # retrain model
    columns, test_data = preprocess_data("TestData_N111_p10.csv")  # load in test data
    test_X, _ = Begin(test_data).initialize()  # discard test responses, don't need them
    U = np.exp(test_X @ B)  # use test X and parameters trained with optimal lambda
    P = U / U.sum(axis=1, keepdims=True)  # new normalized probability matrix
    print(P.shape, P)  # should b 111x5-probability for all classes for each observation
    ordered_probs = P.argsort(axis=1)
    print(ordered_probs)  # show indices sorted in order from least to most probable
    print(P.argmax(axis=1))  # show only the most probable class for observations

    cmap = plt.get_cmap("plasma")  # set a nice colormap, it's perceptually uniform too

    def barplots(op):
        # new list which takes class index and swaps it with corresponding label string
        most_probable = [Begin(data).labels[i] for i in op[:, -1]]
        # data still in order so slice for 5 unknown, 54 Mexican, & 52 African American
        mp_u, mp_mx, mp_aa = (
            most_probable[:5],
            most_probable[5:59],
            most_probable[59:],
        )
        print(mp_u, mp_mx, mp_aa)
        # in order to quickly get & plot the # of observations for each class prediction
        keys_mx, counts_mx = np.unique(mp_mx, return_counts=True)
        keys_aa, counts_aa = np.unique(mp_aa, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, constrained_layout=True, sharex=True, sharey=True
        )  # establish the figure and two subplots

        ax1.set_title(Begin(test_data).labels[1])  # the string "Mexican"
        # plot bars and apply colormap based on y
        plot1 = ax1.bar(keys_mx, counts_mx, color=cmap(counts_mx))
        ax1.bar_label(plot1)  # add labels so we can see exact # for each class
        # f"{Begin(test_data).labels[0][:7]} {Begin(test_data).labels[0][7:]}" works too
        ax2.set_title("African American")  # but this is much easier+cleaner
        plot2 = ax2.bar(keys_aa, counts_aa, color=cmap(counts_aa))
        ax2.bar_label(plot2)

        fig.suptitle("Model results for most probable ancestry label of test data")
        fig.supxlabel("Training labels")
        fig.supylabel("# of observations")
        fig.set_size_inches(12, 7.5)
        return plt.gcf()

    bars_og = barplots(ordered_probs)  # plot the original predictions of most probable
    bars_og.savefig(f"{file_path}/../img/assign3/deliverable4_og.png", dpi=200)

    for i, probs in enumerate(ordered_probs[59:]):  # correction for African Americans
        if probs[4] == 1:  # was curious to see if the most probable label is East Asian
            ordered_probs[59:][i][4] = probs[3]  # then we swap w/ the 2nd most probable

    bars_sb = barplots(ordered_probs)  # plot again, now with substitutions applied
    bars_sb.savefig(f"{file_path}/../img/assign3/deliverable4_sb.png", dpi=200)

    # prepare the array again, this time for donut plots
    most_probable = [Begin(data).labels[i] for i in ordered_probs[:, -1]]
    mp_mx, mp_aa = (most_probable[5:59], most_probable[59:])
    keys_mx, counts_mx = np.unique(mp_mx, return_counts=True)
    keys_aa, counts_aa = np.unique(mp_aa, return_counts=True)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(Begin(test_data).labels[1])
    ax1.pie(
        counts_mx,
        labels=keys_mx,
        autopct="%1.1f%%",
        pctdistance=0.5,
        colors=cmap(np.arange(keys_mx.shape[0]) * 100),
        wedgeprops={"width": 0.35},
    )
    ax2.set_title("African American")
    ax2.pie(
        counts_aa,
        labels=keys_aa,
        autopct="%1.1f%%",
        pctdistance=0.5,
        colors=cmap(np.arange(keys_aa.shape[0]) * 100),
        wedgeprops={"width": 0.35},
    )
    fig.suptitle("Predicted most probable ancestry label, test data+correction applied")
    fig.set_size_inches(12, 7.5)
    plt.savefig(f"{file_path}/../img/assign3/deliverable4_sb_donuts.png", dpi=200)


if __name__ == "__main__":
    main()
