# Feedforward Neural Network for MNIST

from tensorflow import keras
from tensorflow.keras import layers
from interface import MnistClassifierInterface

class FeedForwardMnistClassifier(MnistClassifierInterface):
    # Simple fully-connected neural network for MNIST digits

    def __init__(self, input_dim=784, num_classes=10):
        # Build the neural network layers
        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),      # Flattened 28x28 image input
            layers.Dense(128, activation="relu"),  # First hidden layer
            layers.Dense(64, activation="relu"),   # Second hidden layer
            layers.Dense(num_classes, activation="softmax")  # Output layer for 10 classes
        ])
        # Compile the model with optimizer, loss function and metrics
        self.model.compile(
            optimizer="adam",  # Adaptive optimizer
            loss="sparse_categorical_crossentropy",  # Suitable for integer labels
            metrics=["accuracy"]  # Evaluate model accuracy
        )

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        # Train the feedforward network on the dataset
        # epochs: number of full passes through the dataset
        # batch_size: number of samples per gradient update
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        # Predict class labels for given input
        preds = self.model.predict(X, verbose=0)  # Returns probabilities
        return preds.argmax(axis=1)  # Choose the class with highest probability