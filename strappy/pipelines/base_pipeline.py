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
from sklearn.exceptions import NotFittedError

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

    combined_pipe = Pipeline([("base_pipeline", combined_pipe)])

    if params:
        combined_pipe.set_params(**params)

    return combined_pipe


def apply_pipeline(pipeline:Tuple[str,Pipeline,List[str]],
    df:pd.DataFrame, columns:Union[None,List[str]]=None):
    """
    Apply the steps of a sklearn.pipeline.Pipeline
    to a pandas.DataFrame

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline

    df : pandas.DataFrame

    columns : Union[None, List[str]]

    Returns
    -------
    pandas.DataFrame
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("`pipeline` must be a sklearn.pipeline.Pipeline" +
        f"but received {type(pipeline)}")
    z = df.copy()
    if columns is not None:
        z = z.loc[:,columns]
    try:
        for step in pipeline.steps:
            z = step[1].transform(z)
    except NotFittedError:
        pass
    return z

def apply_column_transformer(col_transformer:ColumnTransformer, df:pd.DataFrame):
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
        try:
            transformed.append(
                (pipeline[0],
                 apply_pipeline(pipeline[1], df.loc[:,pipeline[2]])))
        except:
            pass
    transformed = pd.concat([t[1].reset_index(drop=True) for t in transformed], axis=1)
    return transformed

def transform_dataframe(pipeline:Pipeline, df:pd.DataFrame):
    """
    Apply a Pipeline composed of other Pipelines and
    ColumnTransformer to a pandas.DataFrame

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline

    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    for s in pipeline.steps:
        if isinstance(s[1], Pipeline):
            df = apply_pipeline(s[1],df)
        elif isinstance(s[1], ColumnTransformer):
            df = apply_column_transformer(s[1],df)
    return df


def name_tracker(pipeline:Pipeline, reverse:bool=False):
    """Track names through a sklearn.pipeline.Pipeline
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline

    reverse : bool
        default=False

    Returns
    -------
    dict
        if reversed=False, then {
            name_in_0:[name_in_0_out_0, name_in_0_out_1,...],
            name_in_1:[names_in_1_out_0, names_in_1_out_1],
            ...}
        else: {
            name_in_0_out_0: name_in_0,
            name_in_0_out_1: name_in_0,
            ...}
    """
    names={}
    for s in pipeline.steps:
        if isinstance(s[1], Pipeline):
            _names = _name_tracker_pipeline(s[1])
        elif isinstance(s[1], ColumnTransformer):
            _names = _name_tracker_column_transformer(s[1])
        for k in _names.keys():
            names[k] = list(set(names.get(k,[]) + _names.get(k)))
    if reverse:
        names = {i:k for k,v in names.items() for i in v}
    return names

def _name_tracker_pipeline(pipeline:Pipeline):
    """Helper function for tracking column names through
    a sklearn.pipeline.Pipeline
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline

    Returns
    -------
    names : dict
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("`pipeline` must be a sklearn.pipeline.Pipeline " +
        f"but received {type(pipeline)}")
    try:
        features_in = pipeline.variables_
        names = {i:[i] for i in features_in}
    except:
        return {}
    for step in pipeline.steps:
        try:
            if isinstance(step[1], OneHotEncoder):
                _names = {k:[k + "_" + j for j in v] for k,v in step[1].encoder_dict_.items()}
                for k in _names.keys():
                    names[k] = list(set(names.get(k,[]) + _names.get(k)))
            if isinstance(step[1], AddMissingIndicator):
                _names = {k:[k + "_na"] for k in step[1].variables_}
                for k in _names.keys():
                    names[k] = list(set(names.get(k,[]) + _names.get(k)))
        except:
            pass
    return names

def _name_tracker_column_transformer(col_transformer):
    """Helper function for tracking column names through
    a sklearn.compose.ColumnTransformer
    
    Parameters
    ----------
    col_transformer : sklearn.compose.ColumnTransformer

    Returns
    -------
    names : dict
    """
    if not isinstance(col_transformer, ColumnTransformer):
        raise TypeError("`col_transformer` must be a sklearn.compose.ColumnTransformer " +
        f"but received {type(col_transformer)}")
    names = {}
    for pipeline in col_transformer.transformers_:
        try:
            _names = _name_tracker_pipeline(pipeline[1])
            for k in _names.keys():
                names[k] = list(set(names.get(k,[]) + _names.get(k)))
        except:
            pass
    return names