from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_knn(X_train, X_val, y_train, y_val, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    report = classification_report(y_val, y_pred)
    print(report)

def export_submission(predictions, test_ids, filename):
    """
    Exporteer voorspellingen van een model naar een CSV bestand voor Kaggle.

    Parameters:
        predictions: array met voorspelde waarden
        test_ids: ID-kolom van de testset
        filename: naam van de CSV die wordt aangemaakt
    """
    try:

        submission = pd.DataFrame({
            'id': test_ids,
            'stroke': predictions
        })
        submission.to_csv(filename, index=False)
        print(f"Submission saved as {filename}")

    except Exception as e:
        print(f"Export failed: {e}")