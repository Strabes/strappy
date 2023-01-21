"""
Module of categorical binners
"""

from typing import Union, Optional, List
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalBinnerMixin(TransformerMixin, BaseEstimator): # pylint: disable=too-few-public-methods
    """Mixin for Categorical binners"""
    @staticmethod
    def _check_variables(variables):
        if variables is None:
            return None
        elif isinstance(variables,(str,int)):
            return [variables]
        elif isinstance(variables,(list, tuple)):
            if not all(isinstance(i,(int,str)) for i in variables):
                raise ValueError("All elements of `variables` must be the same type")
            elif len(set(variables)) != len(variables):
                raise ValueError("Duplicate values in variables")
            else:
                return variables

    @staticmethod
    def _check_X(X: pd.DataFrame) -> None:
        if len(set(X.columns.tolist())) != X.shape[1]:
            raise ValueError("Duplicate column names")

    def get_feature_names_out(self) -> list:
        if self.variables_ is not None:
            return self.variables_


    def _check_or_select_variables(self, X):
        if self.variables_ is None:
            variables = []
            for v in X.columns:
                if X[v].dtype in ["object", pd.StringDtype, pd.CategoricalDtype]:
                    variables.append(v)
            self.variables_ = self._check_variables(variables)
        else:
            self._check_X(X)
            if not all([col in self.variables_ for col in X.columns.tolist()]):
                missing = [col for col in X.columns.tolist() if col not in self.variables_]
                raise ValueError(f"`X` is missing the columns {', '.join(missing)}")

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

        for z in self.variables_:
            X.loc[-X[z].isna(),z] = X.loc[-X[z].isna(),z] \
                .map(self.map[z]).fillna(self.other_val)

        return X

class MaxLevelBinner(CategoricalBinnerMixin):        
    """
    MaxLevelBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
                 max_levels = 20, other_val = '_OTHER_'):
        self.variables_ = self._check_variables(variables)
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
        self._check_or_select_variables(X)
        for z in self.variables_:
            cnts = X.groupby(z,dropna=False).size() \
                     .sort_values(ascending = False) \
                     .head(self.max_levels)
            levels = cnts.index.tolist()
            self.map[z] = {l:l for l in levels}
        return self


class PercentThresholdBinner(CategoricalBinnerMixin):
    """
    PercentThresholdBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
        percent_threshold = 0.02, other_val = '_OTHER_'):
        self.variables_ = self._check_variables(variables)
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
        self._check_or_select_variables(X)
        for z in self.variables_:
            cnts = (X.groupby(z,dropna=False).size() / X.shape[0])
            levels = cnts[cnts>=self.percent_threshold].index.tolist()
            self.map[z] = {l:l for l in levels}
        return self


class CumulativePercentThresholdBinner(CategoricalBinnerMixin):
    """
    CumulativePercentThresholdBinner
    """
    def __init__(self, variables: Union[None, int, str, List[Union[str, int]]] = None,
        cum_percent = 0.95, other_val = '_OTHER_'):
        self.variables_ = self._check_variables(variables)
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
        self._check_or_select_variables(X)
        for z in self.variables_:
            cnts = (X.groupby(z,dropna=False).size() / X.shape[0]) \
                      .to_frame(name = z + '_perc').reset_index() \
                      .sort_values([z + '_perc',z], ascending = [False,True]) \
                      .set_index(z) \
                      .cumsum().shift(periods=1, fill_value=0)
            levels = cnts[cnts[z + '_perc']<=self.cum_percent].index.tolist()
            self.map[z] = {l:l for l in levels}
        return self
