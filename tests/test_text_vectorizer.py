"""
Test text vectorization
"""

import pytest
import pandas as pd
from strappy.transformers._text_vectorizer import VectorizeText


@pytest.fixture
def correct_result():
    x = {
        "x_another":{"0":0.0,"1":0.0,"2":1.0,"3":0.0,"4":0.0},
        "x_can":{"0":0.0,"1":0.6141889663,"2":0.0,"3":0.0,"4":0.0},
        "x_for":{"0":0.0,"1":0.6141889663,"2":0.0,"3":0.0,"4":0.0},
        "x_is":{"0":1.0,"1":0.0,"2":0.0,"3":1.0,"4":0.0},
        "x_weekend":{"0":0.0,"1":0.4955237908,"2":0.0,"3":0.0,"4":1.0}
        }
    df = pd.DataFrame(x)
    df.index = pd.to_numeric(df.index, errors='coerce')
    return df


@pytest.fixture
def example_data():
    df = pd.DataFrame({
    'x':[
        "Today is Friday",
        "I can't wait for the weekend",
        "another sentence",
        "tomorrow is saturday",
        "notastopword weekend"],
    'y':[
        "First, I have to work today",
        "Tomorrow will be fun though",
        "why isn't this working?",
        "there appears to be an issue",
        "notastopword issue"]})
    return df

def test_text_vectorizer(example_data,correct_result):
    vt = VectorizeText(params={'max_features':5})
    vals = vt.fit_transform(example_data['x'])
    pd.testing.assert_frame_equal(vals,correct_result)  