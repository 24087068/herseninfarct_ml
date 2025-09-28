import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr

def impossible_values(df: pd.DataFrame):
    """
    Waarden zoeken die hoogst waarschijnlijk onmogelijk of onwaarschijnlijk zijn.
    Retourneerd een dictionary met de problemen.
    """

    problems = {}

    problems['age'] = df[(df['age'] < 0) | (df['age'] > 120)]
    problems['bmi'] = df[(df['bmi'] < 10) | (df['bmi'] > 60)]
    problems['avg_glucose_level'] = df[(df['avg_glucose_level'] < 40) | (df['avg_glucose_level'] > 400)]

    # Boolean checks
    bool_cols = ['heart_disease', 'hypertension', 'work_type_Self-employed',
                'smoking_status_formerly smoked', 'stroke']
    for col in bool_cols:
        problems[col] = df[~df[col].isin([0, 1])]

    return problems

def eda_sum(df: pd.DataFrame, show_heatmap=True):
    """
    EDA samenvatting voor project.

    Parameters:
        df (pd.DataFrame): De input dataset
        show_heatmap (bool): Toon correlatie heatmap
    """
    # Algemene info
    print("---"*33)
    print("Dataset Info")
    print("---"*33)
    print(f"Rijen: {df.shape[0]}, Kolommen: {df.shape[1]}")
    print("Datatypen:")
    print(df.dtypes)
    print("Ontbrekende waarden:")
    missing = df.isnull().sum()
    print(missing)

    # Meta data
    print("---"*33)
    print("Meta data overzicht:")
    print("---"*33)
    df.info()

    # Feature selectie
    selected_features = [
        'age', 'heart_disease', 'hypertension', 'avg_glucose_level',
        'work_type_Self-employed', 'bmi', 'smoking_status_formerly smoked', 'stroke'
    ]
    df_features = df[selected_features].copy()

    # Correlatie analyse
    print("---"*33)
    print("Correlatie Analyse")
    print("---"*33)
    if show_heatmap:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlatie Heatmap")
        plt.show()

    print("---"*33)
    print("Correlaties met 'stroke':")
    print("---"*33)
    correlations = df_features.corr()['stroke'].sort_values(ascending=False)[1:]
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.3f}")

    # Verdeling non-boolean features (ratio niveau)
    print("---"*33)
    print("Verdeling Non-Boolean Features")
    print("---"*33)

    def statistieken_verdeling(df, col):
        """Bereken iqr, skewness en kurtosis voor een kolom."""
        print(f"Statistieken voor {col}:")
        print(df[col].describe())
        print(f"IQR: {iqr(df[col]):.2f}")
        print(f"Skewness: {df[col].skew():.2f}")
        print(f"Kurtosis: {df[col].kurtosis():.2f}")
        sns.histplot(data=df, x=col)
        plt.title(f"Verdeling van {col}")
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x='stroke', y=col)
        plt.title(f"Boxplot van {col} per Stroke")
        plt.show()

    for col in ['age', 'avg_glucose_level', 'bmi']:
        statistieken_verdeling(df_features, col)

    # Verdeling boolean features (nominaal niveau)
    print("---"* 33)
    print("Verdeling Boolean Features")
    print("---"* 33)
    boolean_cols = ['heart_disease', 'hypertension', 'work_type_Self-employed', 'smoking_status_formerly smoked',
                    'stroke']
    for col in boolean_cols:
        print(df_features[col].value_counts())
        print()

    # Categorische kolommen
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print("Categorische Kolommen")
        for col in cat_cols:
            print(f"{col}:")
            print(df[col].value_counts())
            plt.figure(figsize=(6, 8))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Verdeling van {col}")
            plt.show()

    print("---" * 33)
    print("Controle op onmogelijke waarden")
    print("---" * 33)

    problems = impossible_values(df)
    for col, bad_rows in problems.items():
        if not bad_rows.empty:
            print(f"\n Onmogelijke waarden in {col}:")
            print(bad_rows[col])
    print("* EDA FINISHED!!")