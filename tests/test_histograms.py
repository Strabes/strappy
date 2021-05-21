"""
Test histogram plots
To compile baseline images, run:
pytest --mpl-generate-path=tests/baseline
"""

import pytest
import pandas as pd
from strappy.utils.histograms import numeric_histogram, categorical_histogram

@pytest.fixture
def example_data():
    """Data for test"""
    x = [0, 1, 2.2, 1, 3.1,
        -0.23, 1, 2.3, 0, -0.5,
        2, 1.1 ]
    y = list('aaaaabbbccde')
    df = pd.DataFrame({'x':x,'y':y})
    return df

@pytest.mark.mpl_image_compare
def test_numeric_histogram(example_data):
    nh = numeric_histogram(
      example_data,
      x = 'x',
      line_columns = None,
      max_levels = 5,
      stat = 'mean',
      min_levels = 5,
      normalize = True)
    return nh


@pytest.mark.mpl_image_compare
def test_categorical_histogram(example_data):
    nh = categorical_histogram(
        example_data,
        x='y',
        line_columns='x',
        max_levels=3,
        normalize=True)
    return nh