import pytest
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from strappy.pipelines.base_pipeline import (
    create_transformer_pipeline,
    name_tracker
)
import pandas as pd
import numpy as np

input = pd.read_pickle(dir_path + "/create_pipeline_tests/base_pipeline/input.pkl")
results = pd.read_pickle(dir_path + "/create_pipeline_tests/base_pipeline/results.pkl")

def test_base_pipeline():
    p = create_transformer_pipeline(text_cols=['z','w'])
    res = pd.DataFrame(
        p.fit_transform(input),
        columns = name_tracker(p,input).final_cols)
    assert res.equals(results)