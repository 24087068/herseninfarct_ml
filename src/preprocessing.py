import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame):
    """
    Remove impossible values from the dataset.
    Currently only removes BMI > 60.
    """
    df_clean = df.copy()

    # Drop BMI outliers
    df_clean = df_clean[df_clean['bmi'] <= 60]

    # Reset index for consistency
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def normalize_data(train_df, test_df=None, cols_to_scale=None, scaler=None):

    df_train_scaled = train_df.copy()
    if cols_to_scale is None:
        cols_to_scale = train_df.select_dtypes(include=['float64', 'int64']).columns

    exclude_cols = ['id']
    cols_to_scale = [c for c in cols_to_scale if c not in exclude_cols]

    if scaler is None:
        scaler = StandardScaler()
        df_train_scaled[cols_to_scale] = scaler.fit_transform(df_train_scaled[cols_to_scale])
    else:
        df_train_scaled[cols_to_scale] = scaler.transform(df_train_scaled[cols_to_scale])

    df_test_scaled = None
    if test_df is not None:
        df_test_scaled = test_df.copy()
        for col in df_train_scaled.columns:
            if col not in df_test_scaled.columns:
                df_test_scaled[col] = 0
        df_test_scaled = df_test_scaled[df_train_scaled.columns]
        df_test_scaled[cols_to_scale] = scaler.transform(df_test_scaled[cols_to_scale])

    return df_train_scaled, df_test_scaled, scaler

