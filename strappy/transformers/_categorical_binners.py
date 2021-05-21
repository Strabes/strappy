"""
Module of categorical binners
"""

from typing import Union, Optional, List
import pandas as pd
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class CategoricalBinnerMixin: # pylint: disable=too-few-public-methods
    """Mixin for Categorical binners"""
    def transform(self, X : pd.DataFrame):
        """
        Default transform method

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        X = X.copy()

        for z in self.variables:
            X.loc[-X[z].isna(),z] = X.loc[-X[z].isna(),z] \
                .map(self.map[z]).fillna(self.other_val)

        return X

class MaxLevelBinner(CategoricalBinnerMixin,BaseCategoricalTransformer):        
    """
    MaxLevelBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
                 max_levels = 20, other_val = '_OTHER_'):
        self.variables = _check_input_parameter_variables(variables)
        self.max_levels = max_levels
        self.other_val = other_val

    def fit(self, X, y : Optional[pd.Series] = None): # pylint: disable=unused-argument
        """
        Fit method

        Parameters
        ----------
        X : pandas.DataFrame
        """
        self.map = {} # pylint: disable=attribute-defined-outside-init
        X = self._check_fit_input_and_variables(X)
        for z in self.variables:
            cnts = X.groupby(z,dropna=False).size() \
                     .sort_values(ascending = False) \
                     .head(self.max_levels)
            levels = cnts.index.tolist()
            self.map[z] = {l:l for l in levels}
        return self


class PercentThresholdBinner(CategoricalBinnerMixin,BaseCategoricalTransformer):
    """
    PercentThresholdBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
        percent_threshold = 0.02, other_val = '_OTHER_'):
        self.variables = _check_input_parameter_variables(variables)
        self.percent_threshold = percent_threshold
        self.other_val = other_val

    def fit(self, X, y : Optional[pd.Series] = None): # pylint: disable=unused-argument
        """
        Fit method

        Parameters
        ----------
        df : pandas.DataFrame
        """
        self.map = {} # pylint: disable=attribute-defined-outside-init
        X = self._check_fit_input_and_variables(X)
        for z in self.variables:
            cnts = (X.groupby(z,dropna=False).size() / X.shape[0])
            levels = cnts[cnts>=self.percent_threshold].index.tolist()
            self.map[z] = {l:l for l in levels}
        return self


class CumulativePercentThresholdBinner(CategoricalBinnerMixin,BaseCategoricalTransformer):
    """
    CumulativePercentThresholdBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
        cum_percent = 0.95, other_val = '_OTHER_'):
        self.variables = _check_input_parameter_variables(variables)
        self.cum_percent = cum_percent
        self.other_val = other_val

    def fit(self, X, y=None): # pylint: disable=unused-argument
        """
        Fit method

        Parameters
        ----------
        X : pandas.DataFrame
        """
        self.map = {} # pylint: disable=attribute-defined-outside-init
        X = self._check_fit_input_and_variables(X)
        for z in self.variables:
            cnts = (X.groupby(z,dropna=False).size() / X.shape[0]) \
                      .to_frame(name = z + '_perc').reset_index() \
                      .sort_values([z + '_perc',z], ascending = [False,True]) \
                      .set_index(z) \
                      .cumsum().shift(periods=1, fill_value=0)
            levels = cnts[cnts[z + '_perc']<=0.85].index.tolist()
            self.map[z] = {l:l for l in levels}
        return self
