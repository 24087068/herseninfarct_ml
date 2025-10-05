import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

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
    # - Imbalanced learning, 2025, SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    pipeline= Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }
    # - scikit - learn, 2025, StarfieldKFold: https: // scikit - learn.org / stable / modules / generated / sklearn.model_selection.StratifiedKFold.html
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # - scikit-learn, 2025, make_scorer: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
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

#svm
def train_svm(X, y):
    """
    Train een SVM classifier met cross-validation, SMOTE en hyperparameter tuning.

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
        ('svm', SVC())
    ])

    param_grid = {
        'svm__C': [0.05, 0.1, 1,],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': ['scale', 0.1],
        'svm__degree': [2, 3]
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

#decision tree
def train_decision_tree(X, y):
    """
    Train een Decision Tree classifier met cross-validation, SMOTE en hyperparameter tuning.

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
        ('dt', DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        'dt__max_depth': [None, 5, 10, 15, 20],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 3, 4, 5],
        'dt__criterion': ['gini', 'entropy']
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

# Ensemble 1 :
# - ChatGPT, 2025, prompt: Verbeteren van custom ensemble model: https://chatgpt.com/share/68e281a6-b384-800a-8744-210bd4b3c038
def train_random_forest(X, y):
    """
    Train een Random Forest classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__criterion': ['gini', 'entropy']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    print("Random Forest Best Params:", best_params)
    print("Random Forest Best F1 score:", best_score)

    return best_model, best_params, best_score


def train_gradient_boosting(X, y):
    """
    Train een Gradient Boosted Trees classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [3, 5],
        'gb__min_samples_split': [2, 5],
        'gb__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    print("Gradient Boosting Best Params:", best_params)
    print("Gradient Boosting Best F1 score:", best_score)

    return best_model, best_params, best_score


def train_xgboost(X, y):
    """
    Train een XGBoost classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__max_depth': [3, 5],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    print("XGBoost Best Params:", best_params)
    print("XGBoost Best F1 score:", best_score)

    return best_model, best_params, best_score

# Ensemble 2 (Custom)
def train_custom_ensemble(X, y):
    """
    Train een eigen ensemble model met minimaal 3 verschillende classifiers.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): features
        y (pd.Series or np.ndarray): target labels

    Returns:
        best_model: getrained ensemble model
        best_params: dict van beste hyperparameters
        best_score: best cross-validation f1-score
    """

    base_models = [
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier())
    ]

    # Create voting ensemble (soft voting)
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('ensemble', VotingClassifier(estimators=base_models, voting='soft'))
    ])

    # Hyperparameter grid
    param_grid = {
        'ensemble__knn__n_neighbors': [5, 7, 9], # Mischien testen met 8, 10 of 11 erbij en 5 en 7 verwijderen
        'ensemble__knn__weights': ['distance', 'uniform'],
        'ensemble__lr__C': [1, 10, 100],
        'ensemble__rf__n_estimators': [50, 100, 150],
        'ensemble__rf__max_depth': [None, 5, 10]
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