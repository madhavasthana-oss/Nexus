#DEFINE reduce, amplify, subtract_mod, add_mod functions that input(df, columns, factor) and replace the columns columns with 
def reduce(df,columns,ratio):
    """
    Reduces the values in specified columns by a given ratio.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to reduce.
        ratio (float): Factor by which to reduce the values.
        
    Returns:
        pd.DataFrame: DataFrame with reduced values in specified columns.
    """
    
    for col in columns:
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx, f"{col}_reduced", df[col] / ratio)
    df.drop(columns=columns, inplace=True)
    return df

def amplify(df, columns, ratio):
    """
    Amplifies the values in specified columns by a given ratio.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to amplify.
        ratio (float): Factor by which to amplify the values.
        
    Returns:
        pd.DataFrame: DataFrame with amplified values in specified columns.
    """
    
    for col in columns:
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx, f"{col}_amplified", df[col] * ratio)
    df.drop(columns=columns, inplace=True)
    return df

def subtract_mod(df, columns, factor):
    """
    Subtracts a factor from the values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to modify.
        factor (float): Factor to subtract from the values.
        
    Returns:
        pd.DataFrame: DataFrame with modified values in specified columns.
    """
    
    for col in columns:
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx, f"{col}_subtracted", df[col] - factor)
    df.drop(columns=columns, inplace=True)
    return df

def add_mod(df, columns, factor):
    """
    Adds a factor to the values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to modify.
        factor (float): Factor to add to the values.
        
    Returns:
        pd.DataFrame: DataFrame with modified values in specified columns.
    """
    
    for col in columns:
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx, f"{col}_added", df[col] + factor)
    df.drop(columns=columns, inplace=True)
    return df