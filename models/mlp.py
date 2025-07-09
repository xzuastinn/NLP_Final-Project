# models/mlp.py
# ------------------------------------------
# MLP (neural net) for Airbnb sentiment using TF-IDF inputs
# ------------------------------------------
from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train):
    """
    Train a multilayer perceptron with one hidden layer.
    """
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict_mlp(clf, X_test):
    """
    Predict sentiment labels using the trained MLP.
    """
    return clf.predict(X_test)
