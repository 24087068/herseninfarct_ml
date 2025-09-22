import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_sum(df: pd.DataFrame, show_heatmap=True):
    """
    Comprehensive EDA summary.

    Parameters:
        df (pd.DataFrame): Input dataframe
        show_heatmap (bool): Whether to show correlation heatmap for numeric columns
    """

    print("="*50)
    print("* DATAFRAME SHAPE")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("="*50, "\n")

    print("* COLUMN DATA TYPES")
    print(df.dtypes)
    print("\n")

    print("* MISSING VALUES")
    missing = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100
    missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_percent})
    print(missing_df)
    print("\n")

    all_numeric = df.select_dtypes(include='number').shape[1] == df.shape[1]
    print(f"* All columns numeric? {all_numeric}")
    print("="*50, "\n")

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        print("* NUMERIC COLUMN STATISTICS")
        display(df[numeric_cols].describe())
        print("\n")

        if show_heatmap:
            print("* CORRELATION HEATMAP")
            plt.figure(figsize=(8,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.tile("Correlation heatmap")
            plt.show()

        for col in numeric_cols:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print("* CATEGORICAL COLUMN SUMMARY")
        for col in cat_cols:
            print(f"-- {col} --")
            print(df[col].value_counts())
            print("\n")

            plt.figure(figsize(6,8))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Countplot of {col}")
            plt.show()

    print("* EDA FINISHED!!")