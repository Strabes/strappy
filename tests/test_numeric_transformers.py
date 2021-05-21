# import pytest
# import pandas as pd
# import numpy as np

# from strappy.transformers._numeric_transformers import *

# @pytest.fixture
# def example_data():
#     return(pd.DataFrame(
#     {'x':np.linspace(1,50,50),
#      'y':np.linspace(-10,-39,50)}
#     ))

# def test_outlier_percentile_capper(example_data):
#     l = np.quantile(example_data.x.values,0.01)
#     h = np.quantile(example_data.x.values,0.99)
#     z = np.array([l if t < l else t for t in example_data.x.values])
#     z = np.array([h if t > h else t for t in z])
#     res = pd.DataFrame(
#         {'x' : z,
#         'y' : example_data.y.values
#         })
#     res2 = OutlierPercentileCapper(
#         variables='x',lower = 0.01, upper = 0.99).fit_transform(example_data)
#     assert res.equals(res2)