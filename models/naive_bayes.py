# models/naive_bayes.py
# ------------------------------------------
# Multinomial Naive Bayes classifier for Airbnb sentiment
# ------------------------------------------
from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes classifier using the training data.
    """
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def predict_naive_bayes(clf, X_test):
    """
    Predict sentiment labels on test data using trained model.
    """
    return clf.predict(X_test)