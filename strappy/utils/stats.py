import scipy.stats as ss
import numpy as np
from itertools import combinations_with_replacement
import pandas as pd
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

def cramers_corrected_stat(confusion_matrix):
    """
    Calculate Cramers V statistic for categorial-categorial association.
    Uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    See:
    https://stackoverflow.com/questions/20892799/
    using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    
    Parameters
    --------------------------
    confusion_matrix : numpy confusion matrix
        
    Returns
    ---------------------------
    float : Cramer's V statistic with bias correction
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return(np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1))))

def cramers_corrected_matrix(df, reorder_cuthill_mckee = True):
    """
    Calculate Cramers V statistic with bias correction for all
    combinations of columns in pandas DataFrame df
    
    Parameters
    --------------------------
    df : pandas DataFrame - all columns must be categorical
    
    reorder_cuthill_mckee : boolean - whether to reorder to the columns
        based on the reverse Cuthill McKee algorithm applied to the
        matrix of Cramers V statistics
        
    Returns
    ---------------------------
    Z : numpy array with Cramers V statistics
    """
    cols = df.columns.tolist()
    Z = np.zeros((len(cols),len(cols)))
    
    for i,j in combinations_with_replacement(range(len(cols)),2):
        z = cramers_corrected_stat(
                pd.crosstab(df[cols[i]],df[cols[j]]).values
            )
        Z[i,j] = Z[j,i] = z
        
    if reorder_cuthill_mckee is True:
        perm = reverse_cuthill_mckee(
            csr_matrix(Z),
            symmetric_mode = True
        )
        cols = [cols[i] for i in perm]
        gx,gy = np.meshgrid(perm,perm)
        Z = (csr_matrix(Z)[gx,gy]).toarray()
        
    Z = pd.DataFrame(
            Z,
            index = cols,
            columns = cols
        )
    
    return(Z)

def cramers_v(df,x,y):
    if isinstance(x,str): x = np.array([x])
    elif isinstance(x,list): x = np.array(x)
        
    @np.vectorize
    def _cramers_v(x):
        z = cramers_corrected_stat(
                pd.crosstab(df[x],df[y]).values)
        return(z)
    
    return(_cramers_v(x))