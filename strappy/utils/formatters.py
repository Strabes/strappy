import re
import pandas as pd


def space_ticks(ax,max_ticks=5):
    """
    Space ticks out on x axis

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    max_ticks : int

    Returns
    -------
    None
    """
    ticks = ax.get_xticks().tolist()
    labels = ax.get_xticklabels()
    l_space = int(len(labels)/max_ticks)
    ax.set_xticks(ticks[0::(l_space+1)])
    ax.set_xticklabels(labels[0::(l_space+1)])


def fix_legend(ax, bbox_to_anchor=(1.2, 1.05),
        sub_pairs={"^\(Count, ":"","\)$":""}, title = ""):
    """
    Fix legend

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    bbox_to_anchor : tuple 
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [fix_strings(i,sub_pairs=sub_pairs) for i in labels]
    ax.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, title = title)

def fix_strings(s,sub_pairs={"^\(Count, ":"","\)$":""}):
    for i in sub_pairs.items():
        s = re.sub(i[0],i[1],s)
    return s


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