import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

class OneHotComboSparse:
    def __init__(self, cols_to_enc = 'auto', num_cols = 'auto'):
        if isinstance(cols_to_enc,str):
            if cols_to_enc != 'auto': cols_to_enc = [cols_to_enc]
        if isinstance(num_cols,str):
            if num_cols != 'auto': num_cols = [num_cols]
        self.cols_to_enc = cols_to_enc
        self.num_cols = num_cols
        self.fitted = False

    def fit(self, X, y = None):
        self.enc = OneHotEncoder(handle_unknown='ignore')
        if self.cols_to_enc == 'auto':
            dtype_dict = X.dtypes.to_dict()
            self.cols_to_enc = [k for k,v in dtype_dict.items()
                if pd.api.types.is_object_dtype(v)]
        if self.num_cols == 'auto':
            dtype_dict = X.dtypes.to_dict()
            self.num_cols = [k for k,v in dtype_dict.items()
                if pd.api.types.is_numeric_dtype(v)]
        self.enc.fit(X[self.cols_to_enc])
        self.all_preds = self.enc.get_feature_names(
            self.cols_to_enc).tolist() + self.num_cols
        self.fitted = True

    def transform(self, X):
        cat_encoded = self.enc.transform(X[self.cols_to_enc])
        num_to_sparse = sparse.csr_matrix(X[self.num_cols].values)
        X = sparse.hstack([cat_encoded,num_to_sparse])
        return(X)

    def fit_transform(self, X, y = None):
        self.fit(X)
        X = self.transform(X)
        return(X)
