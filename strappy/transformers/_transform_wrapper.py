# """
# TransformWrapper is a generic class for wrapping
# functions that do not have parameter that need to
# be learned from the data. This class implements
# fit, transform and fit_transform methods for the function.
# """

# from ._base import BaseTransformer


# class TransformWrapper(BaseTransformer):
#     """
#     Generic class for wrapping functions that
#     do not have parameters that need to be learned
#     from the data. This class implements fit, transform
#     and fit_transform methods for the function.
#     """

#     def __init__(self, func):
#         """
#         Parameters
#         ----------

#         func : callable
#             A function with a single argument, a pandas.DataFrame,
#             and which returns a pandas.DataFrame
#         """
#         self.func = func
#         self.fitted = False

#     def fit(self, X, y = None):
#         """Fit method
#         Parameters
#         ----------
#         X : pandas.DataFrame
#         """
#         self.fitted = True

#     def transform(self, X):
#         """Transform method
#         Parameters
#         ----------
#         X : pandas.DataFrame
#         """
#         if not self.fitted:
#             raise Exception("Transformation not fit yet")
#         X = X.copy()
#         X = self.func(X)
#         return(X)
