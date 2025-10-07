# Abstract interface for MNIST classifiers

from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    # Defines methods that any MNIST classifier must implement

    @abstractmethod
    def train(self, X_train, y_train):
        # Train the model with training data
        pass

    @abstractmethod
    def predict(self, X_test):
        # Predict class labels for test data
        pass
