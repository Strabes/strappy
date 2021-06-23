"""
Binner functions for grouping numeric variables into
bins
"""

import re
import math
import numpy as np
import pandas as pd


def cutpoints(
    x,
    qntl_cutoff=[0.025,0.975],
    cuts='linear',
    ncuts=10,
    sig_fig=3,
    **kwargs):
    '''
    Function to return cut points and bin labels for a numeric 1-D array

    Parameters
    ----------
    x : numpy 1-D array
        numeric 1-D array

    qntl_cutoff : list
        list of length two with lower and upper quantile cutoffs:
        To prevent extreme outliers from influencing the cutpoints
        for the bins, construct the cutpoints between the qntl_cutoff[0] quantile
        and the qntl_cutoff[1] quantile. If qntl_cutoff is None then do not
        ignore outliers

    cuts: str
        one of: 'linear', 'log', 'logp1', 'quantile'
        'linear' : equally spaced cutpoints
        'log' : logarithmically spaced cutpoints
        'logp1' : logarithmically spaced cutpoints after adding 1
        'root' : apply transform np.sign(x)*np.sqrt(abs(x)) and then
            space points linearly
        'quantile' : cutpoints corresponding to equally spaced quantiles

    ncut : int
        number of cutpoints

    sig_fig : int
        number of significant figures to display in the aesthetically
        printed bin labels

    Returns
    -------
    c_final : numpy 1-D array
        final cut points
    '''

    # Create lower bound:
    lb = np.nanmin(x)
    lb_ord_of_mag = _order_of_mag(lb)
    lb_pwr = sig_fig - 1 - lb_ord_of_mag
    lb = np.floor(lb * 10**lb_pwr) / 10**lb_pwr
    # Create upper bound:
    ub = np.nanmax(x)
    ub_ord_of_mag = _order_of_mag(ub)
    ub_pwr = sig_fig - 1 - ub_ord_of_mag
    ub = np.ceil(ub * 10**ub_pwr) / 10**ub_pwr

    # Apply quantile cutoffs if provided:
    if (qntl_cutoff is not None and
            len(qntl_cutoff) == 2 and
            isinstance(qntl_cutoff[0],float) and
            isinstance(qntl_cutoff[1],float)):
        ep = np.quantile(x, qntl_cutoff)
    else:
        ep = np.array([lb,ub])

    # Create cut points
    if isinstance(cuts,str):
        if cuts == 'linear':
            c = np.linspace(ep[0],ep[1],num = ncuts)
        elif cuts == 'log':
            if ep[0] <= 0:
                msg = "Variable range includes zero when using 'log'" + \
                      " - consider using 'logp1' instead"
                raise ValueError(msg)
            else:
                c = 10**np.linspace(
                    np.log10(ep[0]),
                    np.log10(ep[1]),
                    num = ncuts
                    )
        elif cuts == 'logp1':
            if ep[0] <= -1:
                msg = "Variable range includes zero when using 'log'" + \
                      " - consider using 'root' instead"
                raise ValueError(msg)
            else:
                c = 10**np.linspace(
                    np.log10(ep[0] + 1),
                    np.log10(ep[1] + 1),
                    num = ncuts
                    )
                c = np.sort(np.unique(np.append(0,c)))
        elif cuts ==  'root':
            c = np.linspace(
                np.sign(ep[0]) * np.sqrt(abs(ep[0])),
                np.sign(ep[1]) * np.sqrt(abs(ep[1]))
                )
            c = np.sign(c) * c**2
        elif cuts == 'quantile':
            c = np.quantile(x,np.linspace(0,1,ncuts))
    else:
        # cuts are the actual cut points themselves
        c = cuts

    # add far endpoints to c:
    c = np.unique(np.append(np.append(lb,c),ub))
    # round/format values in c:
    c_ord_of_mag = np.array([_order_of_mag(i) for i in c])
    c_log_rnd = np.round(c / 10.0**c_ord_of_mag, sig_fig - 1)
    c_final = np.unique(c_log_rnd * (10.0**c_ord_of_mag))
    return c_final


def human_readable_num(number, sig_fig=3, **kwargs):
    '''
    Function for making numbers aesthetically-pleasing

    Parameters
    ----------
    number : float or int
        A number to format

    sig_fig : int
        Number of significant figures to print

    Returns
    -------
    z : str
        number formatted as str
    '''
    if np.isnan(number): z = 'MISSING'
    elif number == 0:
        z = '0'
    elif np.abs(number) < 1:
        magnitude = int(np.floor(np.log10(abs(number))))
        # if |number| >= 0.01
        if magnitude >= -2:
            z = ('%.' + str(sig_fig - 1 - magnitude) + 'f') % (number)
            z = _remove_trailing_zeros(z)
        else:    
            final_num = number / 10**magnitude
            #z = ('%.' + str(sig_fig - 1) + 'f%s') % (final_num, 'E' + str(magnitude))
            z = ('%.' + str(sig_fig - 1) + 'f') % (final_num)
            z = _remove_trailing_zeros(z) + 'E' + str(magnitude)
    else:
        units = ['', 'K', 'M', 'G', 'T', 'P']
        k = 1000.0
        magnitude = int(math.floor(math.log(np.abs(number), k)))
        final_num = number / k**magnitude
        if magnitude > 5:
            unit = 'E' + str(int(3*magnitude))
        else:
            unit = units[magnitude]
        if np.abs(final_num) < 10:
            #z = ('%.' + str(sig_fig - 1) + 'f%s') % (final_num, unit)
            z = ('%.' + str(sig_fig - 1) + 'f') % (final_num)
            z = _remove_trailing_zeros(z) + unit
        elif np.abs(final_num) < 100:
            #z = ('%.' + str(sig_fig-2) + 'f%s') % (final_num, unit)
            z = ('%.' + str(sig_fig - 2) + 'f') % (final_num)
            z = _remove_trailing_zeros(z) + unit
        else:
            #z = ('%.' + str(sig_fig-3) + 'f%s') % (final_num, unit)
            z = ('%.' + str(sig_fig - 3) + 'f') % (final_num)
            z = _remove_trailing_zeros(z) + unit

    #z = _remove_trailing_zeros(z)
    return z


def cutter(
    df, x, max_levels=20, point_mass_threshold=0.1,
    sig_fig=3, **kwargs):
    """
    Cut a numeric variable into bins

    Parameters
    ----------
    df : pandas.DataFrame

    x : str
        the name of the numeric variable in 'df' to construct
        bins from

    max_levels : int
        maximum number of bins to create from 'x'

    point_mass_threshold : float
        Levels of 'x' with frequency greater than point_mass_threshold
        get their own bin

    sig_fig : int
        Significant figures to use in binning

    Returns
    -------
    z : pandas.Series
        Categorical series of binned values
    """

    df = df.loc[:,[x]].copy()

    # pm contains any values that exceed point_mass_threshold
    # pm is 1-D numpy.array
    pm = _point_mass(df[x], threshold = point_mass_threshold)

    if len(pm) == 0:
        # if there are no values exceeding point_mass_threshold
        # proceed as usual
        x_no_nan = ~np.isnan(df.loc[:,x].values)
        cps = cutpoints(
            df.loc[x_no_nan,x].values,
            ncuts = max_levels,
            **kwargs)

    elif len(pm) > 0:
        # if there are values exceeding point_mass_threshold
        # put all remaining values in rem
        rem = df.loc[~df[x].isin(pm),[x]]
        x_no_nan = ~np.isnan(rem.loc[:,x].values)
        if len(rem.loc[x_no_nan,x].values) > 0:
            # apply cutpoints to rem if there are non-NaN
            # values
            cps = cutpoints(
                rem.loc[x_no_nan,x].values,
                ncuts = max_levels, # - len(pm),
                **kwargs)
        else:
            # Otherwise, rem has no non-NaN values and
            # we just generate empty cutpoints
            cps = np.array([])

    # Construct bin_labels and pm_labels
    c_final, bin_labels, pm_labels = _finalize_bins(cps,pm,sig_fig=sig_fig)

    # Bin values
    df.loc[~df[x].isin(pm),x + '_BINNED'] = pd.cut(
        df.loc[~df[x].isin(pm),x].values,
        c_final,
        labels=bin_labels,
        include_lowest=True)

    # Bring in point masses
    for i,v in enumerate(pm):
        df.loc[df[x] == v,x + '_BINNED'] = pm_labels[i]

    # Construct final labels
    final_labels = bin_labels+pm_labels
    final_labels.sort()

    # Apply labels
    z = pd.Categorical(
        df.loc[:,x + '_BINNED'].values,
        categories = final_labels)
    return z


def binner_df(
    df, x, new_col=None,
    fill_nan="MISSING", max_levels=20, **kwargs):
    """
    Bin a numeric variable

    Parameters
    --------------------------
    df : pandas.DataFrame

    x : str
        The name of the numeric variable in 'df' to
        construct bins from

    new_col : str
        Use as the name of the binned variable

    fill_nan : str
        Value to fill nans with

    max_levels : int
        Maximum number of bins to create from 'x'

    Returns
    ---------------------------
    pandas.DataFrame including new binned column
    """
    if new_col is None:
        new_col = x
    df_ = df.copy().assign(
        **{new_col: lambda z: cutter(z, x, max_levels, **kwargs)})
    if fill_nan is not None:
        df_.replace({new_col: {np.nan: fill_nan}}, inplace=True)
    return df_


def _log_spcl(x):
    """
    Log special returns the base 10 log of the absolute value of x for
    non-zero x. Otherwise, if x is 0, return 0

    Parameters
    ----------

    x : int or float

    Returns
    -------

    float
    """
    if x == 0:
        return 0
    else:
        return math.log(abs(x), 10)


def _order_of_mag(x):
    """
    Calculate the order of magnitude of a number

    Parameters
    ----------

    x : float or int

    Returns
    -------

    int : order of magnitude of x
    """
    if x == 0:
        ord_of_mag = 0
    else:
        ord_of_mag = int(np.floor(_log_spcl(x)))
    return ord_of_mag


def _point_mass(x, threshold=0.1):
    """
    Find point masses in pandas.Series with frequency exceeding
    specified value

    Parameters
    ----------

    x : pandas.Series

    threshold : float
        If value frequency exceeds threshold, consider value to have
        point mass

    Returns
    -------

    1-D numpy array that contains the point masses
    """
    cnts = x.value_counts(normalize=True)
    v = cnts[cnts > threshold].index.values
    v.sort()
    return v


def _remove_trailing_zeros(num_as_str):
    """
    Remove unnecessary trailing zeros from number

    Parameters
    ----------

    num_as_str : str
        Number as string

    Returns
    -------

    Number as str with unnecessary trailing zeros removed
    """
    if re.search("\\.", num_as_str):
        num_as_str = re.sub("0*$", "", num_as_str)
        num_as_str = re.sub("\\.$", "", num_as_str)
    return num_as_str


def _remove_closest(x, y, exclude_endpoints=True, **kwargs):
    """
    Remove the elements of x that are closest to the elements of y.
    Optionally excluding the endpoints of x in the determination

    Parameters
    ----------
    x : 1-D numpy array

    y : 1-D numpy array

    exclude_endpoints : Boolean
        default True

    Returns
    -------
    x : numpy.array
        numpy 1-D array, the elements of x after
        removing the values closest
        to the elements of y
    """
    x = x.copy()
    if len(x) > 2 or not exclude_endpoints:
        if exclude_endpoints:
            z = x[1:-1]
        else:
            z = x
        r = range(len(z))
        ridx = [min(r, key=lambda i: abs(z[i] - j)) + 1 for j in y]
        ridx = list(set(ridx))
        x = x.tolist()
        for index in sorted(ridx, reverse=True):
            del x[index]
        x = np.array(x)
    return x


def _finalize_bins(x, pm, sig_fig=3, **kwargs):
    """
    Orchestrator for creating bins for numeric variables

    Parameters
    ----------
    x : numpy.array
        1-D array of preliminary bin endpoints

    pm : numpy.array
        1-D array of values with point masses

    sig_fig : int
        Number of significant figures to use

    Returns
    -------
    b : numpy.array
        1-D array of finalizied bin endpoints

    bin_labels : list
        Final bin labels

    pm_labels : list
        Final point mass labels
    """
    b = _remove_closest(x, pm, **kwargs)
    b = np.unique(np.concatenate([b, pm]))
    b.sort()
    bin_labels, pm_labels = _label_constructor(
        b, pm, sig_fig=sig_fig, **kwargs)
    return b, bin_labels, pm_labels


def _label_constructor(x, pm, sig_fig=3, **kwargs):
    """
    Create bin labels for histogramming a numeric variable

    Parameters
    ----------

    x : 1-D numpy array
        The cutpoints for binning

    pm : 1-D numpy array
        The values with point masses

    sig_fig : int
        The number of significant figures to use

    Returns
    -------

    bin_labels : list
        Labels for the bins

    pm_labels : list
        Labels for the point masses
    """
    bin_labels = []
    pm_labels = []
    x_format = [human_readable_num(i, sig_fig=sig_fig) for i in x]
    cntr = 0
    for i in range(len(x)):
        if x[i] in pm:
            pm_labels.append(str(i+cntr+1).zfill(2) + ": " + x_format[i])
            cntr += 1
        if i < len(x) - 1:
            bin_labels.append(
                str(i + cntr + 1).zfill(2) +
                ': ' +
                ('[' if i == 0 and (x[i] not in pm) else '(') +
                x_format[i] +
                ', ' +
                x_format[i+1] +
                (']' if (x[i+1] not in pm) else ')')
            )
    return bin_labels, pm_labels
