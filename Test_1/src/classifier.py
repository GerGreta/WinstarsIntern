# Unified MNIST classifier wrapper

from random_forest import RandomForestMnistClassifier
from feedforward import FeedForwardMnistClassifier
from cnn import CnnMnistClassifier

class MnistClassifier:
    # Wrapper to select one of the MNIST models

    def __init__(self, algorithm):
        # Choose the model based on algorithm string
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier()  # Random Forest
        elif algorithm == "nn":
            self.classifier = FeedForwardMnistClassifier()  # Feedforward NN
        elif algorithm == "cnn":
            self.classifier = CnnMnistClassifier()          # CNN
        else:
            raise ValueError("Unknown algorithm: choose from ['rf', 'nn', 'cnn']")

    def train(self, X_train, y_train):
        # Train the selected model
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        # Predict using the selected model
        return self.classifier.predict(X_test)
