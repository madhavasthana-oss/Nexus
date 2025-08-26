def iqr_quantize(df, columns, n_quantiles=10, multiplier=1.5):
    """
    Bins columns into quantiles after capping outliers using IQR method.
    Outliers are capped at Q1 - multiplier*IQR or Q3 + multiplier*IQR, then quantized.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): Columns to encode.
        n_quantiles (int): Number of quantile bins.
        multiplier (float): IQR multiplier for outlier threshold (default: 1.5).

    Returns:
        pd.DataFrame: DataFrame with columns replaced by quantile bins.
    """
    import numpy as np
    df = df.copy()
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Cap outliers
        capped_values = df[col].clip(lower_bound, upper_bound)
        
        # Create quantile bins
        bins = np.unique(np.quantile(capped_values, np.linspace(0, 1, n_quantiles + 1)))
        quantized_values = np.digitize(capped_values, bins, right=True) - 1
        
        # Store outlier info
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_capped': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]),
            'quantile_bins': bins
        }
        
        # Replace column with quantized version
        col_idx = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        df.insert(col_idx, f"{col}_iqr_quantized", quantized_values)
    
    # Store outlier info as metadata
    df.attrs['iqr_outlier_info'] = outlier_info
    return df


def z_quantize(df, columns, n_quantiles=10, threshold=3):
    """
    Bins columns into quantiles after capping outliers using Z-score method.
    Outliers are capped at mean ± threshold*std, then quantized.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): Columns to encode.
        n_quantiles (int): Number of quantile bins.
        threshold (float): Z-score threshold for outlier detection (default: 3).

    Returns:
        pd.DataFrame: DataFrame with columns replaced by quantile bins.
    """
    import numpy as np
    df = df.copy()
    outlier_info = {}
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # Cap outliers
        capped_values = df[col].clip(lower_bound, upper_bound)
        
        # Create quantile bins
        bins = np.unique(np.quantile(capped_values, np.linspace(0, 1, n_quantiles + 1)))
        quantized_values = np.digitize(capped_values, bins, right=True) - 1
        
        # Store outlier info
        outlier_info[col] = {
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_capped': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]),
            'quantile_bins': bins
        }
        
        # Replace column with quantized version
        col_idx = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        df.insert(col_idx, f"{col}_z_quantized", quantized_values)
    
    # Store outlier info as metadata
    df.attrs['zscore_outlier_info'] = outlier_info
    return df