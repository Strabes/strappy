"""
Test column selector
"""

import pytest
import pandas as pd
from strappy.transformers._variable_selector import MakeColumnSelector

@pytest.fixture
def example_data():
    return(pd.DataFrame(
    {'OneColumn':['a'],
     'AnotherColumn':['a']}
    ))

def test_column_selector(example_data):
    cols = MakeColumnSelector(pattern_exclude="Another")(example_data)
    assert cols == ['OneColumn']