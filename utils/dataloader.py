import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, images, labels, test_size, random_state):
        self.images = images
        self.labels = labels
        self.test_size = test_size
        self.random_state = random_state

    def data_load(self):
        X = np.array(self.images)
        y = np.array(self.labels)
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        return X_train, X_test, y_train, y_test
