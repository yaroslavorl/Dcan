from sklearn import datasets
from sklearn.model_selection import train_test_split
from mxnet import np


def data(bath_size=33):

    X, Y = datasets.load_digits(return_X_y=True)

    X_train, X_test, _, _ = train_test_split(X, Y, train_size=0.9, stratify=Y, random_state=123)

    X_train, X_test = np.array(X_train) / 16., np.array(X_test) / 16.

    X_train = X_train.reshape(bath_size, 49, 1, 8, 8)
    X_test = X_test.reshape(len(X_test), 1, 8, 8)

    return X_train, X_test


