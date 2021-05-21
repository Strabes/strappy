# import numpy as np
# import pandas as pd
# from typing import Union
# from feature_engine.outliers.base_outlier import BaseOutlier


# class OutlierPercentileCapper(BaseOutlier):
    
#     def __init__(self, variables = Union[str,list], lower = 0.01, upper = 0.99):
#         super(OutlierPercentileCapper, self).__init__(variables)
#         self.lower = lower
#         self.upper = upper
#         self.map = {}
        
#     def fit(self, X, y = None):
#         if self.fitted: return
#         for z in self.variables:
#             self.map[z] = {}
#             if self.lower is not None:
#                 self.map[z]['lower'] = X[z].quantile(
#                     self.lower)
#             if self.upper is not None:
#                 self.map[z]['upper'] = X[z].quantile(
#                     self.upper)
#         self.fitted = True
        
#     def transform(self, X, y = None):
#         if not self.fitted:
#             raise Exception("Transformation not fit yet")
#         X = X.copy()
#         for z in self.variables:
#             vals = self.map[z]
#             if 'lower' in vals.keys():
#                 X.loc[X[z] < vals['lower'],z] = vals['lower']
#             if 'upper' in vals.keys():
#                 X.loc[X[z] > vals['upper'],z] = vals['upper']
#         return X