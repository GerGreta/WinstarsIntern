# Random Forest classifier for MNIST

from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    # Wrapper for sklearn's RandomForestClassifier

    def __init__(self, n_estimators=100, random_state=42):
        # Initialize the Random Forest model
        # n_estimators: number of trees in the forest
        # random_state: ensures reproducibility
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            random_state=random_state)

    def train(self, X_train, y_train):
        # Fit the Random Forest on training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Predict class labels for test data
        return self.model.predict(X_test)
