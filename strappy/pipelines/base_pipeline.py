"""
Base pipeline constructs a pipeline of
generic transformations for easy EDA
"""

from typing import Union
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
            p_text.append(("text_" + str(i), VectorizeText(), MakeColumnSelector(pattern=col)))
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


def name_tracker(p, X):
    """
    
    """
    cols_in = X.columns.tolist()
    df = pd.DataFrame({"cols":cols_in,"cols_in":cols_in})

    # indicators for missing numeric cols
    add_missing_ind = p.transformers_[0][1]["add_missing_ind"]
    try:
        nan_num_ind = pd.DataFrame({
            "cols":[i + "_na" for i in add_missing_ind.variables_],
            "cols_in": add_missing_ind.variables_})
        df = pd.concat([df, nan_num_ind])
    except:
        pass

    # onehot encoding of categorical columns
    one = p.transformers_[1][1]["one_hot_encoder"]
    try:
        one_hot_encoder = pd.DataFrame(set().union(*[
            [(k + "_" + i, k) for i in v]
            for k,v in one.encoder_dict_.items()]),
            columns = ["cols", "cols_in"])
        df = pd.concat([df,one_hot_encoder])
    except:
        pass

    # handle the text columns
    running_text_names = []
    for t in p.transformers_[2:-1]:
        try:
            v_name = t[2][0]
            col_tfidf = t[1].vectorizer.get_feature_names()
            col_tfidf_df = pd.DataFrame(
                {"cols": [v_name + "_" + i for i in col_tfidf],
                 "cols_in": [v_name] * len(col_tfidf)})
            df = pd.concat([df,col_tfidf_df])
            running_text_names += [v_name + "_" + i for i in col_tfidf]
        except:
            pass

    numeric_preds = p.transformers_[0][2]
    if len(numeric_preds) > 0:
        final_num_cols = (p
          .transformers_[0][1]
          .transform(
              X.head(1)[numeric_preds])
          .columns.tolist())
    else:
        final_num_cols = []

    object_preds = p.transformers_[1][2]
    if len(object_preds) > 0:
        final_obj_cols = (p
          .transformers_[1][1]
          .transform(X.head(1)[object_preds])
          .columns.tolist())
    else:
        final_obj_cols = []

    df_ = pd.DataFrame({"final_cols":
      final_num_cols + final_obj_cols + running_text_names})

    df = (pd.merge(
           df_, df, left_on="final_cols",
           right_on="cols")
           .loc[:,["final_cols","cols_in"]])

    return df