# image processing

import numpy as np
import matplotlib as mpl
from PIL import Image, ImageStat, ImageOps, ImageDraw, ImageFont

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

def set_axes_size(ax, w, h):
    # ex: imagOp.set_axes_size(grid, 3, 2)
    fig = ax.figure
    aw, ah = np.diff(ax.transAxes.transform([(0, 0), (1, 1)]), axis=0)[0]
    fw, fh = fig.get_size_inches()
    dpi = fig.get_dpi()
    scalew = w / (aw / dpi)
    scaleh = h / (ah / dpi)
    fig.set_size_inches(fw*scalew, fh*scaleh, forward=True)

def set_manual_legend(plam, ax, gap=12, **lgargs):
    # ex: 
    # plam = {
    #     r'$\lambda_\mathrm{D}$':  ({'marker':'o', 'color':'k', 'alpha':0.9, 'linewidth':0, 'mew':0}, 
    #                                {'marker':'v', 'color':'k', 'alpha':0.9, 'linewidth':0, 'mew':0}),
    #     r'$\lambda_\mathrm{R}$':  ({'marker':'o', 'color':'k', 'alpha':0.9, 'linewidth':0, 'fillstyle':'none', 'mew':0.5}, 
    #                                {'marker':'v', 'color':'k', 'alpha':0.9, 'linewidth':0, 'fillstyle':'none', 'mew':0.5})}
    # imagOp.set_manual_legend(plam, ax, ncol=1, loc=3)
    from matplotlib.legend_handler import HandlerLine2D
    import matplotlib.pyplot as plt
    class HandlerXoffset(HandlerLine2D):
        def __init__(self, marker_pad=0.3, numpoints=1, x_offset=0,  **kw):
            HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)
            self._xoffset = x_offset
        def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
            numpoints = self.get_numpoints(legend)

            if numpoints > 1:
                # we put some pad here to compensate the size of the
                # marker
                xdata = np.linspace(-xdescent + self._marker_pad * fontsize,
                                    width - self._marker_pad * fontsize,
                                    numpoints) - self._xoffset
                xdata_marker = xdata
            elif numpoints == 1:
                xdata = np.linspace(-xdescent, width, 2) - self._xoffset
                xdata_marker = [0.5 * width - 0.5 * xdescent - self._xoffset]

    #         print(xdata, self._xoffset)
    #         print(xdata_marker)
            return xdata, xdata_marker
        
    hplot = []
    handlerMaps = {}

    for key, item in plam.items():
        if type(item)==tuple:
            lenHandle = len(item)
            print(plt.plot([], [], **item[0]))
            multipleHandle = [plt.plot([], [], **singleItem)[0] for singleItem in item]
            for i, iHandle in enumerate(multipleHandle):
                handlerMaps[iHandle] = HandlerXoffset(x_offset=(- i + (lenHandle-1)/2)*gap)
            multipleHandle = tuple(multipleHandle)
            hplot.append((multipleHandle, key))
        else:
            singleHandle = plt.plot([], [], **item)[0]
            hplot.append((singleHandle, key))

    handles, labels = zip(*hplot)
    handles = list(handles)
    labels = list(labels)
    ax.legend(handles, labels, handler_map=handlerMaps, **lgargs)


def _get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def _drawTime(imagepath, n, dt):
    font = ImageFont.truetype("Arial.ttf", 48)
    im = Image.open(imagepath)
    w = im.width
    h = im.height
    draw = ImageDraw.Draw(im)
    wt, ht = draw.textsize(str(dt*n)+' ms', font=font)
    draw.text(((w-wt)/2, 0), str(dt*n)+' ms',(255,255, 255), font=font)
    return im

def _get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def caberImageGen(folder, dt, suffix='out', numCho=None):
    # numCho = [0.3, 0.7, 0.8, 0.9],
    # a = caberImageGen(folderename, dt=0.5, numCho = numCho)

    folderImg = glob.glob(folder+'/'+suffix+'*.tif')
    folderImg = sorted(folderImg, key=get_order)
    if numCho is None: 
        pass
    else:
        l = len(folderImg)
        im = _drawTime(folderImg[np.int(numCho[0]*l)], np.int(numCho[0]*l), dt)
        for i in numCho[1:]:
            im2 = _drawTime(folderImg[np.int(i*l)], np.int(i*l), dt)
            im = _get_concat_h(im, im2)
    return im

def ecHollow(df, gbname, palette):
    # grid = sns.scatterplot(data=ampDataOriginFilter,
                            # x=r'$\gamma_0$', y=r'$G^\prime$ (Pa)',
                            # style=r'$c$ (wt%)', hue=r'$c$ (wt%)', palette=seq_col, legend=False, linewidth=0, ax=ax)
    # grid = sns.scatterplot(data=ampDataOriginFilter,
                            # x=r'$\gamma_0$', y=r'$G^{\prime\prime}$ (Pa)',
                            # style=r'$c$ (wt%)', legend=False, ax=ax, fc='none',
                            # edgecolor=ecHollow(ampDataOriginFilter, r'$c$ (wt%)', seq_col))
    keys = list(df.groupby(gbname).groups.keys())
    return df[gbname].map(dict(zip(keys, palette)))

def axMinorFix(ax, subsx=np.linspace(0.1, 0.9, 9), subsy=np.linspace(0.1, 0.9, 9), numticksx=50, numticksy=50):
    locminx = mpl.ticker.LogLocator(base=10.0, subs=subsx, numticks=numticksx)
    ax.xaxis.set_minor_locator(locminx)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    locminy = mpl.ticker.LogLocator(base=10.0, subs=subsy, numticks=numticksy)
    ax.yaxis.set_minor_locator(locminy)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

