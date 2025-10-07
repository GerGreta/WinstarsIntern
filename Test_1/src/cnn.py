# Convolutional Neural Network for MNIST

from tensorflow import keras
from tensorflow.keras import layers
from interface import MnistClassifierInterface

class CnnMnistClassifier(MnistClassifierInterface):
    # CNN for MNIST digit classification

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        # Build CNN architecture
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),        # Input layer (28x28 grayscale)
            layers.Conv2D(32, (3, 3), activation='relu'),  # Conv layer 1
            layers.MaxPooling2D((2, 2)),            # Max pooling 1
            layers.Conv2D(64, (3, 3), activation='relu'),  # Conv layer 2
            layers.MaxPooling2D((2, 2)),            # Max pooling 2
            layers.Flatten(),                        # Flatten for Dense layers
            layers.Dense(64, activation='relu'),     # Fully connected layer
            layers.Dense(num_classes, activation='softmax')  # Output layer
        ])
        # Compile the CNN with optimizer, loss, and accuracy metric
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        # Train CNN on dataset
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        # Predict class labels for images
        preds = self.model.predict(X_test, verbose=0)
        return preds.argmax(axis=1)
