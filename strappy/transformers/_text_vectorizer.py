"""
Vectorize text field
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class VectorizeText(BaseEstimator, TransformerMixin):
    """
    Class for altering sklearn.feature_extraction.text.TfidfVectorizer
    so that its transform method will return a pandas.DataFrame
    """
    def __init__(self, vectorizer = TfidfVectorizer(max_features=100), params : dict = None):
        self.vectorizer = vectorizer
        if params:
            self.vectorizer.set_params(**params)

    def fit(self, X, y = None):
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
        self.vectorizer = self.vectorizer.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform method

        Parameters
        ----------
        X : pandas.Series

        Returns
        -------
        res_df : pandas.DataFrame
        """
        col = X.name
        res = self.vectorizer.transform(X)
        res_df = pd.DataFrame(
            res.todense(),
            columns = [col + "_" + i for i in self.vectorizer.get_feature_names()])
        return res_df