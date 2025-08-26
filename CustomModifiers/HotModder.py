import pandas as pd
import numpy as np
import torch 
from typing import Union,List
from collections import Counter
def hot_encode(df: pd.DataFrame,
               columns: Union[str, List[str]],
               inplace: bool=False) ->pd.DataFrame:
    
    if not inplace:
        df=df.copy()
    if isinstance(columns, str):
        columns=[columns]
    col_dict={}
    for col in columns:
        cc=Counter(df[col])
        for c in cc.keys():
            col_dict[f'{col}_{c}']=df[col].apply(lambda x:1 if x==c else 0)
        df_temp=pd.DataFrame(col_dict,index=df.index)
        df_right=df.iloc[:,df.columns.get_loc(col)+1:]
        df_left=df.iloc[:,:df.columns.get_loc(col)]
        df=pd.concat([df_left,df_temp,df_right],axis=1)

    return df
    