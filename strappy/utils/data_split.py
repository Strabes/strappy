import pandas as pd
import numpy as np
from typing import Union

def data_split(df:pd.DataFrame, train_size:float=0.8, test_size:float=0.1,
    strat_col:Union[str,None]=None, oot_dt_col:Union[str,None]=None,
    oot_dt:Union[str,None]=None, split_col="SPLIT"):

    df = df.copy()
    df.loc[:,split_col] = None
    if oot_dt_col:
        df.loc[
            pd.to_datetime(df[oot_dt_col]) >= pd.to_datetime(oot_dt),
            split_col] = 'OOT'

    if strat_col:
        v = df.loc[lambda x: x[split_col].isnull(),strat_col].unique()
    else:
        v = np.array([i for i,j in enumerate(df[split_col].isnull()) if j])
    
    np.random.seed(42)
    np.random.shuffle(v)
    train_v = np.array(v[:int(train_size*len(v))])
    test_v = np.array(v[int(train_size*len(v)):int((train_size+test_size)*len(v))])
    val_v = np.array(v[int((train_size+test_size)*len(v)):])

    if strat_col:
        df.loc[lambda x: x.apply(
            lambda x: pd.isnull(x[split_col]) and x[strat_col] in train_v, axis=1),
            split_col] = "TRAIN"
        df.loc[lambda x: x.apply(
            lambda x: pd.isnull(x[split_col]) and x[strat_col] in test_v, axis=1),
            split_col] = "TEST"
        df.loc[lambda x: x.apply(
            lambda x: pd.isnull(x[split_col]) and x[strat_col] in val_v, axis=1),
            split_col] = "VAL"
    else:
        df.iloc[train_v,lambda df: df.columns == split_col] = "TRAIN"
        df.iloc[test_v,lambda df: df.columns == split_col] = "TEST"
        df.iloc[val_v,lambda df: df.columns == split_col] = "VAL"

    return df

