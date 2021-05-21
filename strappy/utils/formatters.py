import pandas as pd

def datetime_tester(df, **kwargs):
    """
    Test non-numeric columns in pandas.DateFrame
    """
    dtype_dict = df.dtypes.to_dict()
    
    rel_kw = {key: value for key, value in kwargs.items() 
                if key in pd.to_datetime.__code__.co_varnames}
    
    for k in dtype_dict.keys():
        if not pd.api.types.is_numeric_dtype(dtype_dict[k]):
            try:
                _ = pd.to_datetime(df[k], **rel_kw)
                dtype_dict[k] = _.dtype
            except:
                pass
    return(dtype_dict)