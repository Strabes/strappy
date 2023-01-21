import pytest
import pandas as pd
import numpy as np
from strappy.utils.data_split import data_split

def test_data_split():
    np.random.seed(42)
    df = pd.DataFrame({
        "x":np.arange(10),
        "dt":pd.to_datetime((['2020-01-01']*4 + ['2022-01-01'])*2),
        "y":list("aaabbcddef")})
    z = df.assign(SPLIT = [
        "TRAIN","TRAIN","TRAIN","TEST","TRAIN",
        "TRAIN","VAL","TRAIN","TRAIN","TRAIN"])

    assert z.equals(data_split(df))

    z = df.assign(SPLIT = [
        "TRAIN","TRAIN","TRAIN","TEST","OOT",
        "TRAIN","TRAIN","VAL","TRAIN","OOT"])

    assert z.equals(data_split(df, oot_dt='2021-01-01',oot_dt_col='dt'))