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


def normalize_data(train_df: pd.DataFrame, test_df: pd.DataFrame = None, cols_to_scale=None):
    """
    Normaliseer numerieke features met gebruik van StandardScaler.
    Indien test_df wordt meegegeven, wordt deze automatisch geschaald met dezelfde scaler
    en krijgen train en test exact dezelfde kolommen (missing columns in test worden op 0 gezet).

    Parameters:
        train_df (pd.DataFrame): training dataframe
        test_df (pd.DataFrame, optional): test dataframe
        cols_to_scale (list, optional): kolommen om te schalen; default = alle numerieke

    Returns:
        df_train_scaled (pd.DataFrame): geschaalde train data
        df_test_scaled (pd.DataFrame or None): geschaalde test data (als test_df meegegeven)
        scaler (StandardScaler): fitted scaler voor hergebruik
    """
    df_train_scaled = train_df.copy()
    
    # Determine numeric columns if not specified
    if cols_to_scale is None:
        cols_to_scale = train_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Fit scaler on train
    scaler = StandardScaler()
    df_train_scaled[cols_to_scale] = scaler.fit_transform(df_train_scaled[cols_to_scale])
    
    df_test_scaled = None
    if test_df is not None:
        df_test_scaled = test_df.copy()
        # Align test columns with train columns
        df_test_scaled, df_train_scaled_aligned = df_test_scaled.align(df_train_scaled, join='left', axis=1, fill_value=0)
        # Scale numeric columns in test
        df_test_scaled[cols_to_scale] = scaler.transform(df_test_scaled[cols_to_scale])
        df_train_scaled = df_train_scaled_aligned  # ensure both have exact same columns
    
    return df_train_scaled, df_test_scaled, scaler
