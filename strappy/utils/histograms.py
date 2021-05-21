import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from decimal import Decimal
from .dates import bin_dates
from .binners import (
    cutpoints,
    human_readable_num,
    cutter
)


def plot_bar(p,
            x = 'x',
            line_columns = None,
            normalize = False,
            **kwargs):
    '''
    Function for creating a bar plot for categorical data.
    If input pandas DataFrame includes columns that should be plotted as
    lines, the names of these columns can be passed in 'line_columns'
    By passing a matplotlib figure, 'fig', and axis object, 'ax',
    this function can add the plot to the provided axis
    
    Parameters
    --------------------------
    p : pandas DataFrame object that contains:
        x : the categorical variable to plot along the x-axis
        Count : the height of the bars
        line_columns : optional list of columns to plot as lines

    x : the categorical variable in 'df' to plot along the x-axis

    line_columns : optional list of columns to plot as lines
    
    normalize : Boolean
        If True, convert counts to percents
    
    **kwargs : optional parameters:
        fig : a matplotlib figure
        ax : a matplotlib axis object
        
    Returns
    ---------------------------
    fig : a matplotlib figure
    '''
    if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
        fig = kwargs['fig']; ax = kwargs['ax']
    else:
        fig = plt.figure()
        ax = plt.gca()
        
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
        
    n = p.shape[0]
    if not normalize:
        _stat = '_COUNT_'
        _stat_label = 'Count'
    elif normalize:
        p['_PERCENT_'] = p._COUNT_ / sum(p._COUNT_)
        _stat = '_PERCENT_'
        _stat_label = 'Percent'
        
    _ = ax.bar(
        range(n),
        p.loc[:,_stat],
        align='center',
        width=0.9,
        color = next(prop_iter)['color']
    )
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        p.loc[:,x].values.tolist(),
        rotation=45,ha='right')
    if line_columns is not None:
        twinx = ax.twinx()
        if isinstance(line_columns,str):
            line_columns = [line_columns]
        for i, col in enumerate(line_columns):
            #clr = next(prop_iter)['color']
            _ = twinx.plot(
                range(n),
                p.loc[:,col],
                marker = 'o',
                color = next(prop_iter)['color']
            )
        twinx.spines['top'].set_visible(False)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'], labelpad = 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.sca(ax)
    plt.ylabel(_stat_label)
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    return(fig)




def _numeric_histogram(
    df,
    x = 'x',
    oth_columns = None,
    max_levels = 20,
    stat = 'mean',
    binner = True,
    **kwargs):
    '''
    Function for histogramming a numeric column into bins and
    optionally calculating statistics of other columns within
    these bins
    
    Parameters
    --------------------------
    df : pandas DataFrame object
    
    x : the name of the numeric variable in 'df' to construct bins from
    
    oth_columns : optional list of other columns in 'df' on which to
        calculate 'stat'
        
    max_levels : maximum number of bins to create from 'x'
    
    stat : aggregate statistic to calculate on 'oth_columns' within
        bins of 'x'
        
    Returns
    ---------------------------
    p : pandas DataFrame object
    '''
    if oth_columns is None:
        oth_columns = []
    elif isinstance(oth_columns,str):
        oth_columns = [oth_columns]
    
    #x_grp = x + ' _GROUPED_'
    
    if len(oth_columns) > 0:
        stats = dict(zip(oth_columns,[stat]*len(oth_columns)))
    else:
        stats = dict()
        
    stats['_COUNT_'] = 'sum'
    
    if binner:
        p = (
            df[[*oth_columns,x]].copy()
            .assign(**{x: lambda z: cutter(z,x,max_levels,**kwargs)})
            .replace({x:{np.nan:'MISSING'}})
            .assign(_COUNT_ = 1)
            .groupby(x)
            .agg(stats)
            .reset_index()
            #.rename(columns = {x_grp:x})
            )
    else:
        p = (
            df[[*oth_columns,x]].copy()
            .assign(_COUNT_ = 1)
            .groupby(x,dropna=False)
            .agg(stats)
            .reset_index()
            )
        vals = p[x].unique().tolist()
        vals_format = [str(i+1).zfill(2) +
                       ": " + human_readable_num(j)
                       for i,j in enumerate(vals)]
        p.loc[:,x] = p[x].map(dict(zip(vals, vals_format)))

    return(p)

def _categorical_histogram(
    df,
    x = 'x',
    oth_columns = None,
    max_levels = 20,
    oth_val = '_OTHER_',
    stat = 'mean',
    **kwargs):
    '''
    Function for histogramming a categorical variable into bins and
    optionally calculating statistics of other columns within
    these bins
    
    Parameters
    --------------------------
    df : pandas DataFrame object
    
    x : the name of the categorical variable in 'df' to construct bins from
    
    oth_columns : optional list of other columns in 'df' on which to
        calculate 'stat'
        
    max_levels : maximum number of bins to create from 'x' - the max_level
        values of 'x' with the greatest record counts receive their own levels,
        all other levels are binned as 'oth_val'
        
    oth_val : str used as value for levels with fewer record counts
    
    stat : aggregate statistic to calculate on 'oth_columns' within
        bins of 'x'
        
    Returns
    ---------------------------
    p : pandas DataFrame object
    '''

    if oth_columns is None:
        oth_columns = []
    elif isinstance(oth_columns,str):
        oth_columns = [oth_columns]
    
    x_grp = '_' + x + ' _GROUPED_'
    
    def _max_lvl_cutoff(f):
        y = f[x] if f.RN <= max_levels else oth_val
        return(y)
    
    k = {x_grp: lambda z: z.apply(_max_lvl_cutoff, axis = 1)}
    #k = {x: lambda z: z.apply(_max_lvl_cutoff, axis = 1)}
    cnts = (
        df.groupby(x)
          .size()
          .to_frame(name='_COUNT_')
          .reset_index()
          .sort_values('_COUNT_',ascending = False)
          .assign(RN = lambda x: x['_COUNT_'].rank(method = 'first',ascending = False))
          .assign(**k)
        )
    m = dict(zip(cnts[x],cnts[x_grp]))
    if len(oth_columns) > 0:
        stats = dict(zip(oth_columns,[stat]*len(oth_columns)))
    else:
        stats = dict()
    stats['_COUNT_'] = 'sum'
    p = (
        df.assign(**{x_grp: lambda f: f[x].map(m)})
          .assign(_COUNT_ = 1)
          .groupby(x_grp)
          .agg(stats)
          .reset_index()
          .rename(columns = {x_grp:x})
    )
    return(p)

def numeric_histogram(
    df,
    x = 'x',
    line_columns = None,
    max_levels = 20,
    stat = 'mean',
    min_levels = 20,
    normalize = False,
    **kwargs):
    '''
    Function to create matplotlib histogram plot
    
    Parameters
    --------------------------
    df : pandas DataFrame object
    
    x : the name of the numeric variable in 'df' to construct bins from
    
    line_columns : optional list of other columns in 'df' on which to
        calculate and plot 'stat' within bins of 'x'
        
    max_levels : maximum number of bins to create from 'x'
    
    stat : aggregate statistic to calculate on 'oth_columns' within
        bins of 'x'
        
    min_levels : if 'x' has more than min_levels distinct level,
        induce binning
        
    normalize : Boolean
        If True, transform counts to percents
        
    Returns
    ---------------------------
    p : matplotlib figure

    
    '''
    if 'binner' in kwargs:
        #binner = kwargs['binner']
        pass
    elif len(df[x].unique()) > min_levels:
        kwargs['binner'] = True
    else:
        kwargs['binner'] = False
    p = _numeric_histogram(
        df,
        x = x,
        oth_columns = line_columns,
        max_levels = max_levels,
        stat = stat,
        #binner = binner,
        **kwargs)
    
    p = plot_bar(p,
            x = x,
            line_columns = line_columns,
            normalize = normalize,
            **kwargs)
    return(p)


def categorical_histogram(
    df,
    x = 'x',
    line_columns = None,
    max_levels = 20,
    oth_val = '_OTHER_',
    stat = 'mean',
    normalize = False,
    **kwargs):
    '''
    Function to create matplotlib histogram plot
    
    Parameters
    --------------------------
    df : pandas DataFrame object
    
    x : the name of the categorical variable in 'df' to construct bins from
    
    line_columns : optional list of other columns in 'df' on which to
        calculate and plot 'stat'
        
    max_levels : maximum number of bins to create from 'x' - the max_level
        values of 'x' with the greatest record counts receive their own levels,
        all other levels are binned as 'oth_val'
        
    oth_val : str used as value for levels with fewer record counts
    
    stat : aggregate statistic to calculate on 'oth_columns' within
        bins of 'x'
        
    normalize : Boolean
        If True, use percents instead of counts
        
    Returns
    ---------------------------
    p : matplotlib figure
    '''

    p = _categorical_histogram(
        df,
        x = x,
        oth_columns = line_columns,
        max_levels = max_levels,
        oth_val = oth_val,
        stat = stat,
        **kwargs)
    p = plot_bar(p,
            x = x,
            line_columns = line_columns,
            normalize = normalize,
            **kwargs)
    return(p)
    
    
def categorical_heatmap(
    df,
    x,
    y,
    stat = 'size',
    fillna = 'MISSING',
    width_ratios = [3,1],
    height_ratios = [1,3],
    cmap = 'hot'):
    
    """
    Function for creating bivariate categorical heatmap
    
    Parameters
    -------------------------------
    df : pandas.DataFrame object
    
    x : str
        categorical variable in 'df' to plot along the x-axis
    
    y : str
        categorical variable in 'df' to plot along the y-axis
    
    stat : str
        aggregate function to apply to df after grouping by 'x' and 'y'
    
    fillna : str
        value to fill numpy NaNs with
    
    width_ratios : iterable of length 2
        ratio of the width of the heatmap to the 'y' marginal plot
    
    height_ratios : iterable of length 2
        ratio of the height of the 'x' marginal plot to the heatmap
    
    cmap : str
        name of matplotlib registered colormap
    
    Returns
    -------------------------------
    fig : a matplotlib figure
    """
    
    df2 = df.fillna({x:fillna,y:fillna}) \
        .groupby([x,y]).agg(stat).unstack(0)
    
    fig, axes = plt.subplots(
    nrows = 2,
    ncols = 2,
    sharex = 'col',
    sharey = 'row',
    constrained_layout=True,
    gridspec_kw = {
        'width_ratios' : width_ratios,
        'height_ratios' : height_ratios}
    )

    heatmap = axes[1,0].imshow(df2,aspect='auto',cmap = 'hot');

    axes[1,0].set_xticks(range(len(df2.columns.tolist())));
    axes[1,0].set_xticklabels(df2.columns.tolist(),rotation=45, ha='right');
    axes[1,0].set_xlabel(x);
    axes[1,0].set_yticks(range(len(df2.index.tolist())));
    axes[1,0].set_yticklabels(df2.index.tolist());
    axes[1,0].set_ylabel(y);

    dfx = df.groupby(x).size();
    axes[0,0].bar(range(len(dfx.index.tolist())),dfx.values);


    dfy = df.groupby(y).size();
    axes[1,1].barh(range(len(dfy.index.tolist())),dfy.values);

    axes[0,1].axis('off');

    for ax in [axes[0,0], axes[1,1]]:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_visible(False);

    axes[1,0].spines['top'].set_visible(False);
    axes[1,0].spines['right'].set_visible(False);
    plt.colorbar(heatmap);
    return(fig)


def _stacked_histogram(
    df, x, stack_var, stat = 'count',
    ax = None):
    """
    Create a stacked histogram
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    x : str
        variable for x axis
        
    stack_var : str
        variable to be stacked
        
    stat : str
        'count' or 'percent'
        
    ax : matplotlib axis object
        if None, function will create one
    """
    p = df.loc[:,[x,stack_var]] \
        .groupby([x,stack_var], dropna=False).size()
    if stat == 'percent':
        p = p.groupby(x).transform(lambda x: x/x.sum())
    if ax is None:
        fig = plt.figure(figsize=(12,5))
        ax = fig.gca()
    p = p.stack(1) \
        .plot.bar(ax = ax, stacked = True, width = 0.95);
    p.legend(title = stack_var, bbox_to_anchor = (1.05, 1), loc='upper left');
    plt.xticks(rotation = 45, ha = 'right')
    return(p)

def stacked_dates_histogram(
    df, date_var, cat_var, ax = None,
    title = None, bins = 30, midpoints = True,
    stat = 'count'):
    
    df = df.loc[:,[date_var,cat_var]].copy()
    df.loc[:,date_var] = bin_dates(df.loc[:,date_var], bins, midpoints)
    
    if ax is None:
        fig = plt.figure(figsize=(12,5))
        ax = fig.gca()
        
    p = _stacked_histogram(df, date_var, cat_var, stat = stat, ax = ax)
    
    return(p)