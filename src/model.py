import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_knn(X, y):
    """
    Train een KNN classifier met cross-validation, SMOTE en hyperparameter tuning.

    Parameters:
        X (pd.DataFrame of np.ndarray): features
        y (pd.Series of np.ndarray): target labels

    Returns:
        best_model: getrained model met beste hyperparameters
        best_params: dict van beste hyperparameters
        best_score: best cross-validation f1-score
    """

    pipeline= Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score

# lr
def train_lr(X, y):
    """
    Train een lr classifier met cross-validation, SMOTE en hyperparameter tuning.

    Parameters:
        X (pd.DataFrame of np.ndarray): features
        y (pd.Series of np.ndarray): target labels

    Returns:
        best_model: getrained model met beste hyperparameters
        best_params: dict van beste hyperparameters
        best_score: best cross-validation f1-score
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    param_grid = {
        'lr__C': [0.01, 0.1, 1, 10, 100],
        'lr__penalty': ['l1', 'l2'],
        'lr__solver': ['liblinear']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scorer, n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score