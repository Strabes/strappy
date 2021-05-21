import matplotlib as mpl
import matplotlib.pyplot as plt

def strappy_style():

    colors = {
        'r' : 'r',
        'y' : 'y',
        'b' : 'b',
        'g' : 'g'
    }
    
    plt.style.use({
        'figure.facecolor': 'white',
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.axisbelow': True,
        'image.cmap': 'Greys',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'grid.linestyle': '-',
        'lines.solid_capstyle': 'round',


        'axes.grid': False,
        'axes.facecolor': 'EAEAF2',
        'axes.edgecolor': 'white',
        'axes.linewidth': 0,
        'grid.color': 'white',
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'axes.prop_cycle': mpl.cycler('color', ['008fd5', 'fc4f30', 'e5ae38', '6d904f', '8b8b8b', '810f7c'])
        #'axes.prop_cycle' : mpl.cycler('color',colors)
    })