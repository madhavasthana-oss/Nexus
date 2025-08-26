import pandas as pd
import numpy as np
from typing import Union, List

def binarize(df: pd.DataFrame, 
                                   columns: Union[str, List[str]], 
                                   inplace: bool = False,
                                   handle_negatives: str = 'error',
                                   handle_nan: str = 'zero',
                                   min_bits: int = None) -> pd.DataFrame:
    """
    Improved vectorized version with all features and better performance.
    #explaining the parameters
    Parameters:
    - df: pd.DataFrame - The input DataFrame containing the columns to be processed.
    - columns: Union[str, List[str]] - The column(s) to be converted to binary.
    - inplace: bool - If True, modifies the DataFrame in place; otherwise, returns a new DataFrame.
    - handle_negatives: str - How to handle negative values ('error', 'skip', 'abs').
    - handle_nan: str - How to handle NaN values ('error', 'skip', 'zero').
    - min_bits: int - Minimum number of bits to use for binary representation.
    Returns:
    - pd.DataFrame - The DataFrame with the specified columns converted to binary representation.
    """
    if not inplace:
        df = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
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
        
        # Handle negative values
        if (df[col] < 0).any():
            if handle_negatives == 'error':
                raise ValueError(f"Column '{col}' contains negative values")
            elif handle_negatives == 'skip':
                print(f"Skipping column '{col}' due to negative values")
                continue
            elif handle_negatives == 'abs':
                df[col] = df[col].abs()
        
        # Use PyTorch for vectorized bit extraction
        import torch
        data = torch.tensor(df[col].astype(int).values)
        max_value = int(torch.max(data).item())

        if max_value == 0:
            num_bits = 1
        else:
            num_bits = max_value.bit_length()

        if min_bits is not None:
            num_bits = max(num_bits, min_bits)

        bit_positions = torch.arange(num_bits)
        bit_matrix = ((data.unsqueeze(1) >> bit_positions) & 1).numpy()
        
        # Get column position and drop original
        col_idx = df.columns.get_loc(col)
        df.drop(columns=[col], inplace=True)
        
        # Insert all binary columns efficiently
        new_cols = {}
        for i in range(num_bits):
            new_cols[f'{col}_{i}'] = bit_matrix[:, i]
        
        # Create temporary DataFrame and concat for better performance
        temp_df = pd.DataFrame(new_cols, index=df.index)
        
        # Split the original DataFrame and insert the new columns
        left_part = df.iloc[:, :col_idx]
        right_part = df.iloc[:, col_idx:]
        
        df = pd.concat([left_part, temp_df, right_part], axis=1)
    
    return df

