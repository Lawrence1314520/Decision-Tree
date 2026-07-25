import numpy as np


def load_data(filepath):

    # why it is not used pd.read_csv()?
    # np.loadtxt read the whole csv file into 2D numpy array
    # and it is necessary to use data.to_numpy() later if using np.read_csv()
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :-1] # this mean get all columns except the last one
    y = data[:, -1].astype(int) # this mean only get the last column, convert it to integers

    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred)