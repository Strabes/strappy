import pytest
import pandas as pd
import numpy as np
import math

from strappy.utils.binners import (
    cutpoints,
    human_readable_num,
    cutter,
    binner_df,
    _log_spcl,
    _order_of_mag
)

def test_cutpoints():
    x = np.array([1.213,43,9.32,4.22324,-1.6,5.2321,32,0.123])
    z = np.array([-1.6, -1.3, 19.9, 41.1, 43. ])
    y = cutpoints(x,ncuts=3,sig_fig=3)
    assert np.array_equal(z,y)

def test_human_readable_num():
    assert human_readable_num(20.321) == '20.3'
    assert human_readable_num(20321) == '20.3K'


@pytest.fixture
def example_data():
    return(
        pd.DataFrame(
           {'x':[
               0, 1, 2.2, 1, 3.1,
               -0.23, 1, 2.3, 0, -0.5,
               2, 1.1 ]}
    ))

@pytest.fixture
def example_data_binned():
    c = pd.Categorical(
        ['02: 0', '04: 1', '05: (1, 2.98]',
        '04: 1', '06: (2.98, 3.1]', '01: [-0.5, 0)',
        '04: 1', '05: (1, 2.98]', '02: 0',
        '01: [-0.5, 0)', '05: (1, 2.98]',
        '05: (1, 2.98]'],
        categories = [
            '01: [-0.5, 0)', '02: 0', '03: (0, 1)',
            '04: 1', '05: (1, 2.98]','06: (2.98, 3.1]',
            'MISSING'])
    return(c)    

def test_cutter(example_data, example_data_binned):
    x = pd.Series(cutter(example_data,'x',3))
    print(x)
    print(pd.Series(example_data_binned))
    assert x.equals(pd.Series(example_data_binned))

def test_binner_df(example_data, example_data_binned):
    df = example_data.rename(columns = {'x':'xy'})
    z = df.join(pd.DataFrame(example_data_binned)).rename(columns = {0:'wz'})
    df = binner_df(df,'xy','wz',max_levels=3)
    print(df)
    print(z)
    assert df.equals(z)

def test_log_spcl():
    assert _log_spcl(0) == 0
    assert math.log(abs(-2.34),10) == _log_spcl(-2.34)

def test_order_of_mag():
    assert _order_of_mag(0) == 0
    assert _order_of_mag(123456) == 5