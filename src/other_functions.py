from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

def evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=1):
    """
    Train een KNN-classifier en evalueer deze met behulp van een confusion matrix, precisie, recall en f1-score.
    Parameters:
        X_train, X_test, y_train, y_test: trainings- en testdatasets
        n_neighbors (int): aantal buren voor KNN
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))