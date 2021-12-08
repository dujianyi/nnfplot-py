### Environment for using sns under the default settings ###
# Oct 10, 2020

# plot
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# data
import pandas as pd
from pandas import DataFrame, Series

# numerical operation 
import numpy as np

# reload lib
import importlib

# image processing
from PIL import Image, ImageStat, ImageOps

# numba
from numba import jit

# warnings 
import warnings
warnings.filterwarnings('ignore')

# Environment

# Apply the default theme
params = {'figure.figsize':        (6, 4),
          'xtick.labelsize':       16,
          'ytick.labelsize':       16,
          'axes.labelsize':        16,
          'font.sans-serif':       'Arial',
          'lines.linewidth':       1.5,
          'axes.linewidth':        1,
          'xtick.minor.visible':   True,
          'ytick.minor.visible':   True,
          'xtick.direction':       'in',
          'ytick.direction':       'in',
          'xtick.major.width':     1,
          'ytick.major.width':     1,
          'legend.frameon':        False,
          'legend.fontsize':       12,
          'axes.grid':             False,
          'axes.formatter.limits': (0, 0),
          'axes.formatter.use_mathtext': True,
          'axes.formatter.useoffset':    False,
          'axes.formatter.limits': (-2, 3),
          'figure.dpi':            300,
          }

sns.set_theme(style='ticks', rc=params)
plt.rcParams.update(params)


# customized markers

import matplotlib.path as mpath
import matplotlib.patches as patches

_circle = mpath.Path.circle()
_circle_inner = mpath.Path.circle(radius=0.7)
mkC = mpath.Path(_circle.vertices, _circle.codes)
mkHC = mpath.Path(np.concatenate([_circle.vertices, _circle_inner.vertices[::-1]]), 
                  np.concatenate([_circle.codes, _circle_inner.codes]))

# customized packages

import fileOp
import dataOp
import imagOp
# import foamOp