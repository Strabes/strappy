"""
Test dates
"""

import pytest
import pandas as pd
from strappy.utils.dates import bin_dates

@pytest.fixture
def example_data():
    d = '''
    {"dt":{"0":"2021-03-07",
    "1":"2020-12-19","2":"2021-01-20",
    "3":"2020-04-24","4":"2020-03-31",
    "5":"2021-10-19","6":"2020-06-26",
    "7":"2020-11-29",
    "8":"2021-04-06","9":"2020-02-10"}}
    '''

    df = pd.read_json(d).assign(dt = lambda df : pd.to_datetime(df.dt))
    return df

@pytest.fixture
def correct_result():
    d = '''
    {"dt":{"0":"2021-04-16",
    "1":"2020-12-14","2":"2020-12-14",
    "3":"2020-04-11","4":"2020-04-11",
    "5":"2021-08-18","6":"2020-08-12",
    "7":"2020-12-14","8":"2021-04-16",
    "9":"2020-04-11"}}
    '''
    df = pd.read_json(d).assign(dt = lambda df : pd.to_datetime(df.dt))

    return df

def test_bin_dates(example_data,correct_result):
    res = (bin_dates(example_data.dt,bins=5)
            .to_frame()
            .assign(dt = lambda df: pd.to_datetime(df.dt)))

    pd.testing.assert_frame_equal(res,correct_result)