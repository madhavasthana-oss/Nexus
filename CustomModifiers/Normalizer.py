import pandas as pd
from typing import Union, List

def l1_normalize(df: pd.DataFrame,
               columns: Union[str, List[str]],
               inplace: bool = False) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame to a range of [0, 1].

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (Union[str, List[str]]): Column(s) to normalize.
        inplace (bool): If True, modifies the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    if not inplace:
        df = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        col_sum = df[col].sum()
        if col_sum == 0:
            raise ValueError(f"Column '{col}' has a sum of zero, cannot normalize.")
        col_idx = df.columns.get_loc(col)
        df[col] = df[col] / col_sum
        df.rename(columns={col:f'{col}_l1_normalized'},inplace=True)
    return df

def l2_normalize(df: pd.DataFrame,
               columns: Union[str, List[str]],
                inplace: bool = False) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame to a range of [0, 1] using L2 normalization.
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (Union[str, List[str]]): Column(s) to normalize.
        inplace (bool): If True, modifies the DataFrame in place. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    if not inplace:
        df = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        col_norm = (df[col] ** 2).sum() ** 0.5
        if col_norm == 0:
            raise ValueError(f"Column '{col}' has a norm of zero, cannot normalize.")
        col_idx = df.columns.get_loc(col)
        df[col] = df[col] / col_norm
        df.rename(columns={col:f'{col}_l2_normalized'},inplace=True)
    return df

def z_score_normalize(df, columns=None, inplace=False, handle_nan='error', min_bits=None):
    """
    z-score the specified columns in a DataFrame by converting them to binary format.
    This function standardizes the columns by subtracting the mean and dividing by the standard deviation.
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to standardize (default: None, which means all columns)
    - inplace: if True, modifies the DataFrame in place (default: False)
    - handle_nan: how to handle NaN values ('error', 'skip', 'zero', 'mean', 'median')
    - min_bits: minimum number of bits to use for binary representation (default: None)
    
    Returns:
    - pandas DataFrame with standardized columns
    """
    import pandas as pd
    import numpy as np
    
    if not inplace:
        df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Warning: Column '{col}' not found in DataFrame")
            continue
        
        # Handle NaN values
        if df[col].isnull().any():
            if handle_nan == 'error':
                raise ValueError(f"Column '{col}' contains NaN values")
            elif handle_nan == 'skip':
                print(f"Skipping column '{col}' due to NaN values")
                continue
            elif handle_nan == 'zero':
                df[col] = df[col].fillna(0)
            elif handle_nan == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif handle_nan == 'median':
                df[col] = df[col].fillna(df[col].median())
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            raise ValueError(f"Column '{col}' has zero standard deviation, cannot standardize")
        df[col] = (df[col] - mean) / std
        df.rename(columns={col:f'{col}_z_score_normalized'},inplace=True)
    return df

def stable_z_score_normalize(df: pd.DataFrame,
                 columns: Union[str, List[str]],
                 inplace: bool = False) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame to a range of [0, 1] using batch normalization.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (Union[str, List[str]]): Column(s) to normalize.
        inplace (bool): If True, modifies the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    if not inplace:
        df = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    import numpy as np
    # generate a small epsilon value to avoid division by zero
    eps = np.random.uniform(1e-8, 1e-6)
    for col in columns:
        mean = df[col].mean()
        var = df[col].var()
        df[col] = (df[col] - mean) / (var + eps) ** 0.5
        df.rename(columns={col:f'{col}_stable_z_score_normalized'},inplace=True)
    return df