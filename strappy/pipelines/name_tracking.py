from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import (
    #TransformerMixin,
    OneToOneFeatureMixin,
    #ClassNamePrefixFeaturesOutMixin,
    BaseEstimator
    )
import re
import itertools
from sklearn.preprocessing import OneHotEncoder
from feature_engine.encoding import RareLabelEncoder

#OneToOneFeatureMixin_Ext = (OneToOneFeatureMixin, OneHotEncoder, RareLabelEncoder)

def name_tracker_columntransformer(columntransformer):
    if not isinstance(columntransformer, ColumnTransformer):
        raise TypeError("`columntransformer` must be a sklearn.compose.ColumnTransformer" +
            f" but got type {type(columntransformer)}")
    feature_names_in = sorted(list(columntransformer.feature_names_in_), reverse=True)
    feature_names_out = sorted(list(columntransformer.get_feature_names_out()), reverse=True)
    name_map = {n : [] for n in feature_names_in}
    for transformer in columntransformer.transformers_:
        if isinstance(transformer[1], Pipeline):
            names = name_tracker_pipeline(transformer[1])
        elif isinstance(transformer[1], ColumnTransformer):
            names = name_tracker_columntransformer(transformer[1])
        # elif isinstance(transformer[1], OneToOneFeatureMixin):
        #     names = name_tracker_onetoonefeaturemixin_ext(transformer[1])
        elif isinstance(transformer[1], BaseEstimator):
            names = name_tracker_baseestimator(transformer[1])
        for key in names.keys():
            name_map[key] = [transformer[0] + "__" + n for n in names[key]]
    missing = []
    accounted_for = set().union(*list(name_map.values()))
    for name_out in feature_names_out:
        if name_out not in accounted_for:
            missing.append(name_out)
    if len(missing) > 0:
        raise Exception(f"Couldn't account for outputs: {', '.join(missing)}")
    return name_map

def name_tracker_baseestimator(baseestimator):
    if not isinstance(baseestimator, BaseEstimator):
        raise TypeError(f"`baseestimator` must be a {BaseEstimator}" +
            f" but got type {type(baseestimator)}")
    feature_names_in = sorted(list(baseestimator.feature_names_in_), reverse=True)
    feature_names_out = sorted(list(baseestimator.get_feature_names_out()), reverse=True)
    if len(feature_names_in) == 1:
        return {feature_names_in[0] : feature_names_out}
    elif all([False if re.match(i[0],i[1]) else True
              for i in list(itertools.product(feature_names_in, feature_names_out))]):
        return {f: feature_names_out for f in feature_names_in}
    else:
        return match_features_in_and_out(feature_names_in,feature_names_out)

        

def match_features_in_and_out(feature_names_in, feature_names_out):
    feature_names_in = sorted(list(feature_names_in), reverse=True)
    feature_names_out = sorted(list(feature_names_out), reverse=True)
    name_map = {n: [] for n in feature_names_in}
    for name_out in feature_names_out:
        _matched = False
        for name_in in feature_names_in:
            if re.match(name_in, name_out):
                _matched = True
                name_map[name_in].append(name_out)
                break
        if not _matched:
            raise Exception(f"Didn't find matching input column for {name_out}")
    return name_map

# def name_tracker_onetoonefeaturemixin_ext(onetoonefeaturemixin_ext):
#     if not isinstance(onetoonefeaturemixin_ext, OneToOneFeatureMixin_Ext):
#         raise TypeError(f"`onetoonefeaturemixin` must be a {OneToOneFeatureMixin_Ext}" +
#             f" but got type {type(onetoonefeaturemixin_ext)}")
#     name_map = match_features_in_and_out(
#         onetoonefeaturemixin_ext.feature_names_in_,
#         onetoonefeaturemixin_ext.get_feature_names_out())
#     return name_map



def name_tracker_pipeline(pipeline):
    if not isinstance(pipeline, Pipeline):
        raise TypeError("`pipeline` must be a sklearn.pipeline.Pipeline" +
            f" but got type {type(pipeline)}")
    feature_names_in = list(pipeline.feature_names_in_)
    feature_names_out = list(pipeline.get_feature_names_out())
    name_map = {n : [n] for n in feature_names_in}
    for step in pipeline.steps:
        if isinstance(step[1], Pipeline):
            # handle pipeline
            names = name_tracker_pipeline(step[1])
        elif isinstance(step[1], ColumnTransformer):
            # handle ColumnTransformer
            names = name_tracker_columntransformer(step[1])
        # elif isinstance(step[1], OneToOneFeatureMixin_Ext):
        #     names = name_tracker_onetoonefeaturemixin_ext(step[1])
        elif isinstance(step[1], BaseEstimator):
            names = name_tracker_baseestimator(step[1])
        for key in names.keys():
            for nm_key, nm_v in name_map.items():
                if key in nm_v:
                    name_map[nm_key] = list(set(nm_v + names[key]))
    missing = []
    accounted_for = set().union(*list(name_map.values()))
    for name_out in feature_names_out:
        if name_out not in accounted_for:
            missing.append(name_out)
    if len(missing) > 0:
        raise Exception(f"Couldn't account for outputs: {', '.join(missing)}")
    for key in name_map.keys():
        name_map[key] = [v for v in name_map[key] if v in feature_names_out]
    return name_map