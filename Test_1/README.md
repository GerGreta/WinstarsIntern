# MNIST Classifier Project

This project implements **three models** for MNIST digit classification using **OOP**:

- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network

All models implement the `MnistClassifierInterface` and are wrapped in the `MnistClassifier` class for unified training and prediction.

---

## Project Structure
```
Test_1/
│
├─ src/
│ ├─ interface.py
│ ├─ random_forest.py
│ ├─ feedforward.py
│ ├─ cnn.py
│ └─ classifier.py
│
├─ notebook/
│ └─ mnist_classifiers.ipynb
│
├─ requirements.txt
└─ README.md
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GerGreta/Winstars_Intern.git
cd Winstars_Intern
```
2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the environment:
Windows PowerShell: .\.venv\Scripts\Activate.ps1
Bash/Linux/Mac: source .venv/bin/activate

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Using the Jupyter Notebook
```bash
jupyter notebook
```
Open notebook/mnist_classifiers.ipynb and run the cells to see:
- Training of Random Forest, Feed-Forward NN, and CNN
- Accuracy metrics
- Visualization of predictions

### 2. Using the Python Classes Directly
```python
from classifier import MnistClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
clf = MnistClassifier("cnn")  # options: "rf", "nn", "cnn"

# Train and predict
clf.train(X_train.values.reshape(-1,28,28,1), y_train)
preds = clf.predict(X_test.values.reshape(-1,28,28,1))
```

## How It Works
1. MnistClassifierInterface defines abstract methods train() and predict().
2. RandomForestMnistClassifier uses sklearn.ensemble.RandomForestClassifier.
3. FeedForwardMnistClassifier uses keras.Sequential with dense layers.
4. CnnMnistClassifier uses keras.Sequential with Conv2D and MaxPooling layers.
5. MnistClassifier is a wrapper that selects one of the three models based on a string parameter ("rf", "nn", "cnn").

## Notes

- All code is well-commented for clarity.
- Notebook blocks include visualizations of predictions.
- The pipeline is fully reproducible with the provided random states.
