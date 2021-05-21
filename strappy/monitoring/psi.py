"""
Module for calculating Population Stability Index
"""

from typing import Union
import numpy as np
import pandas as pd


def _validate_variables(
    df : pd.DataFrame = None,
    variables : Union[str,list] = None):
    """Validate that `df` is a `pandas.DataFrame` and
    `variables` is a subset of columns"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be instance of pandas.DataFrame")
    cols = df.columns.tolist()
    cols_to_str = ", ".join(cols)
    if variables is None:
        return cols
    elif isinstance(variables, str):
        if not variables in cols:
            raise ValueError(f"{variables} not in `df.columns`: {cols_to_str}")
        else:
            return [variables]
    elif isinstance(variables, list):
        missing_vars = [i for i in variables if i not in cols]
        if len(missing_vars) > 0:
            missing_vars = ", ".join(missing_vars)
            raise ValueError(f"{missing_vars} not in `df.columns`: {cols_to_str}")
        else:
            return variables
    else:
        t = str(type(variables))
        raise TypeError(f"`variables` must be str or list, but found {t}")

def _psi(e : np.ndarray, o : np.ndarray, k = 1e-6):
    assert e.shape == o.shape
    assert len(e.shape) == 1
    if k > 0:
        e = np.maximum(e,k)
        o = np.maximum(o,k)
    psi = np.dot((o - e) , np.log(o / e))
    return psi

def _get_frequencies(df : pd.DataFrame = None, cols : list = None):
    dist = {}
    for c in cols:
        d = df.loc[:,c].value_counts(normalize = True, dropna = False).to_dict()
        dist[c] = d
    return dist

def _dicts_to_df(d1, d2, fillna = 0):
    d1 = pd.DataFrame.from_dict(d1, orient='index').sort_index()[0]
    d2 = pd.DataFrame.from_dict(d2, orient='index').sort_index()[0]
    d = pd.concat([d1,d2],axis=1).fillna(fillna)
    return d



class PSI:
    """Class for calculating Population Stability Index"""
    def __init__(self):
        self._cols = None
        self._dist = None

    def fit(self, df : pd.DataFrame = None, variables : Union[str,list] = None):
        """Fit method
        Parameters
        ----------
        df : pandas.DataFrame

        variables : Union[str,list]
        """
        df = df.copy()
        cols = _validate_variables(df, variables)
        self._cols = cols
        self._dist = _get_frequencies(df, cols)

    def transform(self, df : pd.DataFrame):
        df = df.copy()
        _cols = _validate_variables(df, self._cols)
        dist = _get_frequencies(df, self._cols)
        res = {}
        for col in self._cols:
            d = _dicts_to_df(self._dist[col],dist[col])
            psi = _psi(d.iloc[:,0].values, d.iloc[:,1].values)
            res[col] = psi
        return res
