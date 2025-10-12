import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

def train_knn(X, y, X_test=None):
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
        'knn__n_neighbors': [7, 8],
        'knn__weights': ['distance'],
        'knn__metric': ['euclidean'] # euclidean used to run better, force it to use now to understand
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

    test_preds = None
    if X_test is not None:
        final_model = KNeighborsClassifier(
            n_neighbors=best_params['knn__n_neighbors'],
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds

# lr
def train_lr(X, y, X_test=None):
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
        ('lr', LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42))
    ])

    param_grid = {
        'lr__C': [0.003, 0.005, 0.008, 0.01],
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

    test_preds = None
    if X_test is not None:
        final_model = LogisticRegression(
            class_weight='balanced',
            max_iter=3000,
            C=best_params['lr__C'],
            penalty=best_params['lr__penalty'],
            solver='liblinear',
            random_state=42
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds

#svm
def train_svm(X, y, X_test=None):
    """
    Train een SVM classifier met cross-validation, SMOTE en hyperparameter tuning.

    Parameters:
        X (pd.DataFrame of np.ndarray): features
        y (pd.Series of np.ndarray): target labels
        kernel (str): 'poly' of 'linear'

    Returns:
        best_model: getrained model met beste hyperparameters
        best_params: dict van beste hyperparameters
        best_score: best cross-validation f1-score
    """

    # Switched out regular SVM for LinearSVC as regular SVM took to long
    base_svm = LinearSVC(dual=False, max_iter=10000, class_weight='balanced', random_state=42)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('svm', calibrated_svm)
    ])

    param_grid = {
            'svm__estimator__C': [0.005, 0.01, 0.02, 0.03]
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    test_preds = None
    if X_test is not None:
        final_model = LinearSVC(
            C=best_params['svm__estimator__C'],
            dual=False,
            max_iter=10000,
            class_weight='balanced',
            random_state=42
        )
        final_model_calibrated = CalibratedClassifierCV(
            estimator=final_model, method='sigmoid', cv=3
        )
        final_model_calibrated.fit(X, y)
        test_preds = final_model_calibrated.predict(X_test)

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds

#decision tree
def train_decision_tree(X, y, X_test=None):
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
        'dt__max_depth': [4, 5, 6],
        'dt__min_samples_split': [2, 3],
        'dt__min_samples_leaf': [1, 2],
        'dt__criterion': ['entropy']
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

    test_preds = None
    if X_test is not None:
        final_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=best_params['dt__max_depth'],
            min_samples_split=best_params['dt__min_samples_split'],
            min_samples_leaf=best_params['dt__min_samples_leaf'],
            criterion='entropy'
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds

# Ensemble 1 :
# - ChatGPT, 2025, prompt: Verbeteren van custom ensemble model: https://chatgpt.com/share/68e281a6-b384-800a-8744-210bd4b3c038
def train_random_forest(X, y, X_test=None):
    """
    Train een Random Forest classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [100, 150, 200],
        'rf__max_depth': [4, 5, 10, 15],
        'rf__min_samples_split': [2, 3],
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

    test_preds = None
    if X_test is not None:
        final_model = RandomForestClassifier(
            random_state=42,
            n_estimators=best_params['rf__n_estimators'],
            max_depth=best_params['rf__max_depth'],
            min_samples_split=best_params['rf__min_samples_split'],
            min_samples_leaf=best_params['rf__min_samples_leaf'],
            criterion=best_params['rf__criterion'],
            n_jobs=-1
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Random Forest Best Params:", best_params)
    print("Random Forest Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds


def train_gradient_boosting(X, y, X_test=None):
    """
    Train een Gradient Boosted Trees classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__learning_rate': [0.05, 0.1, 0.15],
        'gb__max_depth': [2, 3, 4],
        'gb__min_samples_split': [2, 3],
        'gb__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    test_preds = None
    if X_test is not None:
        final_model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=best_params['gb__n_estimators'],
            learning_rate=best_params['gb__learning_rate'],
            max_depth=best_params['gb__max_depth'],
            min_samples_split=best_params['gb__min_samples_split'],
            min_samples_leaf=best_params['gb__min_samples_leaf']
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Gradient Boosting Best Params:", best_params)
    print("Gradient Boosting Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds


def train_xgboost(X, y, X_test=None):
    """
    Train een XGBoost classifier met cross-validation, SMOTE en hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    param_grid = {
        'xgb__n_estimators': [150, 200, 250],
        'xgb__learning_rate': [0.01, 0.02, 0.05],
        'xgb__max_depth': [2, 3, 4],
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

    test_preds = None
    if X_test is not None:
        final_model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_estimators=best_params['xgb__n_estimators'],
            learning_rate=best_params['xgb__learning_rate'],
            max_depth=best_params['xgb__max_depth'],
            subsample=best_params['xgb__subsample'],
            colsample_bytree=best_params['xgb__colsample_bytree'],
            n_jobs=-1
        )
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("XGBoost Best Params:", best_params)
    print("XGBoost Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds

# Ensemble 2 (Custom)
def train_custom_ensemble(X, y, X_test=None):
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

    # Replace older SVC with LinearSVC for improved speed and better F1
    base_svm = LinearSVC(dual=False, max_iter=10000, class_weight='balanced', random_state=42)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)

    base_models = [
        ('lr', LogisticRegression(max_iter=2000, solver='liblinear')),
        ('rf', RandomForestClassifier()),
        ('svm', calibrated_svm)
    ]

    # Create voting ensemble (soft voting)
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('ensemble', VotingClassifier(estimators=base_models, voting='soft'))
    ])

    # Hyperparameter grid
    param_grid = {
        'ensemble__lr__C': [0.01, 0.1, 1, 10],
        'ensemble__rf__n_estimators': [100, 150],
        'ensemble__rf__max_depth': [5, 10],
        'ensemble__svm__estimator__C': [0.05, 0.1, 0.2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    test_preds = None
    if X_test is not None:
        final_svm = LinearSVC(
            dual=False,
            max_iter=10000,
            class_weight='balanced',
            C=best_params['ensemble__svm__estimator__C'],
            random_state=42
        )
        final_calibrated_svm = CalibratedClassifierCV(estimator=final_svm, method='sigmoid', cv=3)

        final_lr = LogisticRegression(
            max_iter=2000,
            solver='liblinear',
            C=best_params['ensemble__lr__C'],
            random_state=42
        )

        final_rf = RandomForestClassifier(
            n_estimators=best_params['ensemble__rf__n_estimators'],
            max_depth=best_params['ensemble__rf__max_depth'],
            random_state=42
        )

        final_models = [
            ('lr', final_lr),
            ('rf', final_rf),
            ('svm', final_calibrated_svm)
        ]

        final_model = VotingClassifier(estimators=final_models, voting='soft')
        final_model.fit(X, y)
        test_preds = final_model.predict(X_test)

    print("Best Params:", best_params)
    print("Best F1 score:", best_score)

    return best_model, best_params, best_score, test_preds