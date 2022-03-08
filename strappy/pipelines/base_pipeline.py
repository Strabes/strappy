"""
Base pipeline constructs a pipeline of
generic transformations for easy EDA
"""

from typing import Union, Tuple, List
import numpy as np
import pandas as pd

from feature_engine.imputation import (
    ArbitraryNumberImputer,
    AddMissingIndicator,
    CategoricalImputer
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)

from ..transformers._variable_selector import MakeColumnSelector
from ..transformers._text_vectorizer import VectorizeText

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_transformer_pipeline(params=None, text_cols : Union[None,list]=None):
    """
    Pipeline for preprocessing transformations

    Parameters
    ----------
    params : dict
        dictionary of parameters for pipeline

    text_cols : Union[None, list]

    Returns
    -------
    combined_pipe : sklearn.pipeline.Pipeline
        A pipeline to fit
    """
    p_num = Pipeline([
        ("add_missing_ind", AddMissingIndicator()),
        ("arb_num_imputer", ArbitraryNumberImputer(arbitrary_number=0))
    ])

    p_cat = Pipeline([
        ("cat_imputer", CategoricalImputer(fill_value= "_MISSING_")),
        ("rare_label_enc", RareLabelEncoder()),
        ("one_hot_encoder", OneHotEncoder())
    ])

    p_text = []
    if text_cols is not None:
        pattern_exclude = "^" + "$|^".join(text_cols) + "$"
        for i, col in enumerate(text_cols):
            p_text.append(
                ("text_" + str(i),
                Pipeline([("text_" + str(i), VectorizeText())]),
                MakeColumnSelector(pattern=col)))
    else:
        pattern_exclude = None

    transformers = [
        ('num', p_num, MakeColumnSelector(dtype_include=np.number)),
        ('cat', p_cat, MakeColumnSelector(
            dtype_include=object,pattern_exclude = pattern_exclude))
    ] + p_text

    combined_pipe = ColumnTransformer(transformers, remainder='drop')

    if params:
        combined_pipe.set_params(**params)

    return combined_pipe


def apply_pipeline(pipeline:Tuple[str,Pipeline,List[str]], df):
    """
    Apply the steps of a sklearn.pipeline.Pipeline
    to a pandas.DataFrame

    Parameters
    ----------
    pipeline : Tuple(str, sklearn.pipeline.Pipeline, List[str])

    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    if not isinstance(pipeline[1], Pipeline):
        raise TypeError("`pipeline[1]` must be a sklearn.pipeline.Pipeline" +
        f"but received {type(pipeline[1])}")
    z = df.copy().loc[:,pipeline[2]]
    for step in pipeline[1].steps:
        z = step[1].transform(z)
    return z

def transform_dataframe(col_transformer, df):
    """
    Transform a pandas.DataFrame using a sklearn.compose.ColumnTransformer

    Parameters
    ----------
    col_transformer : sklearn.compose.ColumnTransformer

    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    if not isinstance(col_transformer, ColumnTransformer):
        raise TypeError("`col_transformer` must be a sklearn.compose.ColumnTransformer" +
        f"but received {type(col_transformer)}")
    transformed = []
    for pipeline in col_transformer.transformers_:
        transformed.append(
            (pipeline[0],
             apply_pipeline(pipeline, df)))
    transformed = pd.concat([t[1] for t in transformed], axis=1)
    return transformed