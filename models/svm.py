# models/svm.py
# ------------------------------------------
# Support Vector Machine classifier for Airbnb sentiment
# ------------------------------------------
from sklearn.svm import LinearSVC

def train_svm(X_train, y_train):
    """
    Train a linear SVM classifier.
    """
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    return clf

def predict_svm(clf, X_test):
    """
    Predict sentiment labels on test data.
    """
    return clf.predict(X_test)