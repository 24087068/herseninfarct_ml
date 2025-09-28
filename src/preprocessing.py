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


def normalize_data(df: pd.DataFrame, cols_to_scale=None):
    """
    normaliseer numerieke features met gebruik van StandardScaler.
    
    Parameters:
        df (pd.DataFrame): input dataframe
        cols_to_scale (list): kollomen om te schalen, default = alle numerieke
    
    Returns:
        df_scaled (pd.DataFrame): dataframe met geschaalde waarden
        scaler (StandardScaler): fitted scaler (voor later gebruik op de test set)
    """
    df_scaled = df.copy()

    if cols_to_scale is None:
        cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled, scaler
