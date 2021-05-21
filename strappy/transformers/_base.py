# import numpy as np
# import pandas as pd
# from typing import Union, Optional
# from sklearn.base import BaseEstimator, TransformerMixin

# class BaseTransformer(BaseEstimator, TransformerMixin):
#     """
#     Base class for all transformers
#     """
    
#     def __init__(self, variables: Union[str,list]):
#         if isinstance(variables,str): variables = [variables]
#         self.variables = variables
#         self.fitted = False
        
#     def _reset(self):
#         self.fitted = False
        
#     def _check_if_fit(self):
#         return(self.fitted)
    
#     def _validate_variables(self, X, variables, dtypes):
#         if len(variables) == 1:
#             self._validate_one(X, variables, dtypes)
#         else:
#             for z in variables:
#                 self._validate_one(X, z, dtypes)
                
#     def fit_transform(self, X : pd.DataFrame, y : Optional[pd.Series] = None):
#         """
#         Apply fit and transform methods
        
#         Parameters
#         ----------
#         X : pandas.DataFrame

#         y : pandas.Series
#             Optional

#         Returns
#         -------
#         pandas.DataFrame
#         """
#         self.fit(X)
#         return self.transform(X)
        
#     def _validate_one(self, X, variable, dtypes):
#         if not variable in X.columns:
#             raise ValueError(variable + " is not a column in the DataFrame")
#         else:
#             self._validate_one_dtype(X, variable, dtypes)

            
#     def _validate_one_dtype(self, X, variable, dtypes):
#         if not X[variable].dtypes in dtypes:
#             dtypes_str = ", ".join([str(t) for t in dtypes])
#             raise TypeError(
#                 variable + " is not one of the excepted dtypes: " + dtypes_str)
#         else:
#             pass