"""
Test stats module
"""

import pytest
import pandas as pd
from strappy.utils.stats import (
    cramers_corrected_stat,
    cramers_corrected_matrix,
    cramers_v)

@pytest.fixture
def example_data():
    """Data for test"""
    x = list('aaaaabbbccde')
    y = list('abcdeabcabde')
    z = list('aaaaabbbzzzz')
    df = pd.DataFrame({'x':x,'y':y,'z':z})
    return df


def test_cramers_corrected_matrix(example_data):
    ans = pd.DataFrame({
        'z': [1.0, 0.881917, 0.0],
        'x': [0.881917, 1.0, 0.0],
        'y': [0.0, 0.0, 1.0]},
        index=['z', 'x', 'y'])
    res = cramers_corrected_matrix(example_data)
    pd.testing.assert_frame_equal(res,ans)
