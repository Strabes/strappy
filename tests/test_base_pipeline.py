import pytest
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from strappy.pipelines.base_pipeline import (
    create_transformer_pipeline,
    #transform_dataframe
)
import pandas as pd
import numpy as np

input = pd.read_pickle(dir_path + "/create_pipeline_tests/base_pipeline/input.pkl")
results = pd.read_pickle(dir_path + "/create_pipeline_tests/base_pipeline/results.pkl")

def test_base_pipeline():
    p = create_transformer_pipeline(text_cols=['z','w'])
    p = p.fit(input)
    p.set_output(transform="pandas")
    res = p.transform(input)
    #res = transform_dataframe(p,input)
    res.equals(results)
    #pd.testing.assert_frame_equal(res,results)