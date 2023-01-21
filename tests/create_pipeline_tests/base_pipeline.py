import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from strappy.pipelines.base_pipeline import (
    create_transformer_pipeline,
    transform_dataframe
)
import pandas as pd
import numpy as np

N = 100

phrases = [
    'test text','more test text','python is cool',
    'enjoy your weekend','that movie was ok','go to the moon',
    'the movie was cool','more money more problems',"let's go",
    'see you soon']

np.random.seed(42)
df = pd.DataFrame({
    'x' : [1.2,3,np.nan,7,-3,np.nan,1.2383,0.32,-50,300] * int(N/10),
    'y' : np.random.choice(
        ['a'] + [np.nan] + list("bcdefghi"),
        size=N, p=[0.3,0.2,0.1,0.1,0.1,0.05,0.05,0.05,0.025,0.025]),
    'z': np.random.choice(phrases,
        size=N, p=[0.3,0.2,0.1,0.1,0.1,0.05,0.05,0.05,0.025,0.025]),
    'w' : np.random.choice(phrases,
        size=N, p=[0.3,0.2,0.1,0.1,0.1,0.05,0.05,0.05,0.025,0.025])
})

p = create_transformer_pipeline(text_cols=['z','w'])

p = p.fit(df)

res = transform_dataframe(p, df)

df.to_pickle(dir_path + "/base_pipeline/input.pkl")
res.to_pickle(dir_path + "/base_pipeline/results.pkl")