### Environment for using sns under the default settings ###
# Oct 10, 2020

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import importlib
from PIL import Image, ImageStat, ImageOps

# Environment

# Apply the default theme
params = {'figure.figsize':        (6, 4),
          'xtick.labelsize':       18,
          'ytick.labelsize':       18,
          'axes.labelsize':        18,
          'font.sans-serif':       'Arial',
          'lines.linewidth':       1.5,
          'axes.linewidth':        1,
          'xtick.minor.visible':   False,
          'ytick.minor.visible':   False,
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
          'axes.formatter.limits': (-2, 2),
          'figure.dpi':            250,
          
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

# I/O tools

def reIndex(df):
    # recombine 'row 1 (row 2)' as new header including units
    df.columns = df.columns.map(lambda h: '{} ({})'.format(h[0], h[1]))
    return df


def readTxtDelimiter(fileName, delimiter='[step]', title=[0], header=[1,2]):
    # read a huge text file with delimiter to separate sheets, and output to a new xlsx file; title and header correspond to sheet name and header row numbers 
    import os.path
    from io import StringIO 
    if os.path.exists(fileName+'.xlsx'):
        print(fileName+'.xlsx file exists. Do nothing.')
    else:
        print(fileName+'.xlsx file does not exist. Creating now...')
        writer = pd.ExcelWriter(fileName+'.xlsx')
        with open(fileName+'.txt') as fp:
            contents = fp.read()
            for entry in contents.split(delimiter):
                stringData = StringIO(entry)
                df = pd.read_csv(stringData, sep="\t", skiprows=title, header=header)
                df.to_excel(writer, entry.split('\n')[title[0]+1])
        writer.save()
        
        
def readXLS(fileName, reindex=False, appTag=None, *args, **kwargs):
    # read from a xls sheet with the same arguments as read_excel
    import os.path
    if os.path.exists(fileName+'.xlsx') or os.path.exists(fileName+'.xls'):
        if os.path.exists(fileName+'.xlsx'):
            p = pd.read_excel(fileName+'.xlsx', *args, **kwargs)
        else:
            p = pd.read_excel(fileName+'.xls', *args, **kwargs)
        
        if reindex:
            p = reIndex(p)
            
        if appTag is not None: 
            for tag, value in appTag.items():
                p[tag] = value
            
        return p
        
    else:
        print('No '+fileName+'.xls file. Please run readTxtDelimiter(fileName) first or generate from the original software.')

        
# Data processing


def addData(pd_in, loc, val):
    # return as x=, y= - derectly plot
    return pd_in.iloc[:, loc] + val


def scaleData(pd_in, loc, val):
    # return as x=, y= - derectly plot
    return pd_in.iloc[:, loc]*val


def truncData(pd_in, xrange, loc=0):
    # return pd
    return pd_in[(pd_in.iloc[:, loc] >= xrange[0]) & (pd_in.iloc[:, loc] <= xrange[1])].dropna()


def refLine(fun, xrange=None, xin=None):
    if xin is None:
        x = np.linspace(xrange[0], xrange[1], 100)
        return x, fun(x)
    else:
        return xin, fun(xin)
    
from lmfit import Minimizer, Parameters, minimizer, Parameter
    
def fitLine(initParams, fitPack=None, fitRange=None, logFit=False, minFun=None):
    # data is dataFrame, x, y, minFun is list of str/functions to fit correspondingly
    def objective(params):
        res = []
        for i, (data, ix, iy, f) in enumerate(fitPack):
            datacopy = data.dropna(subset=[ix, iy])
            if fitRange is not None:
                datacopy = datacopy[(datacopy[ix]>=fitRange[i][0]) & (datacopy[ix]<=fitRange[i][1])]
            if logFit:
                datacopy = datacopy[datacopy[iy]>0]
                res.append(np.log(f(params, datacopy[ix]))-np.log(datacopy[iy]))
            else:
                res.append(f(params, datacopy[ix])-datacopy[iy])
        res = np.array(res).flatten()
        # res[res>1e308] = 1e20
        # res[np.isnan(res)] = 1e20
        return res
        
    if minFun is None:
        minner = Minimizer(objective, initParams)
    else:
        # Directly given minFun
        minner = Minimizer(minFun, initParams)
        
    print('Start fitting...')
    
    try:
        result = minner.minimize(method='leastsq')
        print('Finished fitting...')
        return result
    except TypeError: 
        print('Error: all data seem to be NaN...')
        raise

    #np.sum(result.residual)


# image processing
def generateImageSeq(imgs, stamps=None, stampsColor=(255, 255, 255), stampsSize=64, split=0, scaleBar=None, orient=1):
    # to generate Img sequence, with optional stamps on the top
    # split specifies the can be something in the middle
    # orietation 0 is horizontal, 1 is vertical
    # scaleBar (10, 100, '100um') - 10um/px, 100um, marked as "100um"
    # example: generateImageSeq(items, stamps=stamps, split=10)
    imgSize = list(np.shape(Image.open(imgs[0])))
    finImgSize = imgSize.copy()
    # finImgSize[orient] = finImgSize[orient]*(len(imgs)) + split*(len(imgs)-1)
    finImgSize[orient] = 0
    splitSize = imgSize.copy()
    if split > 0:
        splitSize[orient] = split
        splitImg = np.ones(splitSize)*255
    else:
        splitImg = None
    finImg = np.empty(finImgSize)
    fnt = ImageFont.truetype('Arial.ttf', stampsSize)
    
#     "Pillow/Tests/fonts/FreeMono.ttf",
    for i, imagei in enumerate(imgs):
        pic = Image.open(imagei)
        if stamps is not None:
            ImageDraw.Draw(pic).text(
                (imgSize[0]/20, imgSize[1]/40),  # Coordinates
                stamps[i],  # Text
                fill=stampsColor,  # Color
                font=fnt,
            )
        if scaleBar is not None:
            ImageDraw.Draw(pic).line(
                (imgSize[0]*5/6, imgSize[1]*23/40, imgSize[0]*5/6+scaleBar[1]/scaleBar[0], imgSize[1]*23/40),
                fill=stampsColor,
                width=5,
            )
            ImageDraw.Draw(pic).text(
                (imgSize[0]*4/5, imgSize[1]*25/40),  # Coordinates
                scaleBar[2],  # Text
                fill=stampsColor,  # Color
                font=fnt,
            )
        if (splitImg is not None) and (i > 0):
            finImg = np.append(finImg, splitImg, axis=orient)
        finImg = np.append(finImg, np.array(pic), axis=orient)
    return Image.fromarray(np.uint8(finImg))
