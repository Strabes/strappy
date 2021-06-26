"""
Vectorize text field
"""

#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class VectorizeText(TfidfVectorizer):
    """
    Class for altering sklearn.feature_extraction.text.TfidfVectorizer
    so that its transform method will return a pandas.DataFrame
    """
    def fit(self, X, y=None):
        """
        Fit method

        Parameters
        ----------
        X : pandas.Series

        y : array_like

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:,0]
        self.colname = X.name
        return super(VectorizeText, self).fit(X, y)

    def transform(self, X, y=None):
        """
        Transform method

        Parameters
        ----------
        X : pandas.Series

        Returns
        -------
        res_df : pandas.DataFrame
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:,0]
        res = super(VectorizeText,self).transform(X)
        colname = self.colname
        res_df = pd.DataFrame(
            res.todense(),
            columns = [colname + "_" + i for i in self.get_feature_names()])
        return res_df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)