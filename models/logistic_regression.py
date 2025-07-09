from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, balanced=True):
    """
    Train a logistic-regression classifier.
    Uses L2 regularization by default.
    If balanced=True, inverse-frequency class weights are applied.
    """
    clf = LogisticRegression(
        penalty="l2",
        max_iter=1000,
        class_weight="balanced" if balanced else None,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return clf

def predict_logistic_regression(clf, X_test):
    return clf.predict(X_test)