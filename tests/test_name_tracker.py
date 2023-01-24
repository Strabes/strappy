import pytest
import pandas as pd
import numpy as np
from strappy.pipelines.name_tracking import name_tracker_pipeline
from strappy.pipelines.base_pipeline import create_transformer_pipeline

def test_name_tracker_pipeline():
    input_df = pd.DataFrame({
        "x" : [1.2, 3, np.nan, 7.2, 2.1, 4.2, 8, 1.3, -1, 2.6],
        "y" : ["a"]*4 + ["b"]*3 + [None] + ["c"] + ["d"],
        "z" : ["This is some text."]*4 + ["This is some more text"] + ["Python is cool!"]*3 + [""] + ["None"]
    })
    correct_names_tracked = {'x': ['num__x', 'num__x_na'],
                             'y': ['cat__y_a', 'cat__y_b', 'cat__y_Rare'],
                             'z': ['text_0__z_is',
                             'text_0__z_python',
                             'text_0__z_cool',
                             'text_0__z_some',
                             'text_0__z_this',
                             'text_0__z_text',
                             'text_0__z_more',
                             'text_0__z_none']}
    p = create_transformer_pipeline(text_cols=["z"],
            params={"base_pipeline__cat__rare_label_enc__tol":0.15,
                    "base_pipeline__cat__rare_label_enc__n_categories":2})
    p.fit(input_df)
    names_tracked = name_tracker_pipeline(p)
    assert names_tracked.keys() == correct_names_tracked.keys()
    for key in names_tracked.keys():
        assert set(names_tracked[key]) == set(correct_names_tracked[key])