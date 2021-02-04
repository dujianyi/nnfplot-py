# to calculate radius, only in pixels but not considering resolution

import numpy as np
from scipy import ndimage, signal, interpolate
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob
from scipy.signal import savgol_filter

def normalizeGray(frame):
    sh = np.shape(frame)
    output = frame
    try:
        output = np.sum(output, axis=2)
    except:
        pass
    output = output*1.0/np.max(output.flatten())
    return output

def logCalc(img, sigma, fsize):
    siz   = (sigma-1)/2
    std2   = fsize**2
    x = np.arange(-siz, siz+1, 1)
    xv, yv = np.meshgrid(x, x)
    arg = -(xv**2 + yv**2)/(2*std2)
    h = np.exp(arg)
    # h(h<eps*max(h(:))) = 0;
    sumh = np.sum(h)
    if not sumh == 0:
        h = h/sumh
    h1 = np.multiply(h, (xv**2 + yv**2 - 2*std2)/(std2**2))
    h = h1 - np.sum(h1)/sigma
    h = h - np.sum(h)/np.size(h)
    return ndimage.convolve(img, h, mode='nearest')
    
def extractRadius(frame, fsize=2, thresh=0.0005):
    
    sigma = 2*fsize*3+1
    img = normalizeGray(frame)
    result = ndimage.gaussian_laplace(img, sigma=fsize)
    # result = logCalc(img, sigma, fsize)

    s1 = np.where((result[:, :-1]<0) & (result[:, 1:]>0) & (np.abs(result[:, :-1]-result[:, 1:])>thresh))
    img_e = np.zeros(np.shape(img))
    img_e[s1[0], s1[1]] = 1
    CC, CCnum = measure.label(img_e, connectivity=2, return_num=True)
    numPixels = [np.size(np.where(CC==i+1)) for i in range(CCnum)]
    ordr = np.argmax(numPixels)
    d1 = np.where(CC==ordr+1)
    sort1 = np.argsort(d1[0])
    x1 = d1[0][:]
    y1 = savgol_filter(d1[1][:], min(81, int((len(x1)-1) // 2 * 2 + 1)), 2)
#     plt.figure(figsize = (30,30)); plt.imshow(frame)
#     plt.plot(d1[1], d1[0])
#     plt.plot(y1, x1)
    
    s2 = np.where((result[:, :-1]>0) & (result[:, 1:]<0) & (np.abs(result[:, :-1]-result[:, 1:])>thresh))
    img_e = np.zeros(np.shape(img))
    img_e[s2[0], s2[1]] = 1
    CC, CCnum = measure.label(img_e, connectivity=2, return_num=True)
    numPixels = [np.size(np.where(CC==i+1)) for i in range(CCnum)]
    ordr = np.argmax(numPixels)
    d2 = np.where(CC==ordr+1)
    sort2 = np.argsort(d2[0])
    x2 = d2[0][:]
    y2 = savgol_filter(d2[1][:], min(81, int((len(x2)-1) // 2 * 2 + 1)), 2)
#     plt.plot(d2[1], d2[0])
#     plt.plot(y2, x2)
    
    dist = [[(i1-i2)**2+(j1-j2)**2 for i2, j2 in zip(x2, y2)] for i1, j1 in zip(x1, y1)]
    minDC = np.unravel_index(np.argmin(dist, axis=None), np.shape(dist))
    
    return np.sqrt(dist[minDC[0]][minDC[1]]), x1[minDC[0]], y1[minDC[0]]+1, x2[minDC[1]], y2[minDC[1]]+1

def processFolder(folder, fps, dPlate=6, skipFile=1):
    
    def orderFileName(frames):
        import re 
        import math
        from pathlib import Path 

        file_pattern = re.compile(r'.*?(\d+).*?')
        def get_order(file):
            match = file_pattern.match(Path(file).name)
            if not match:
                return math.inf
            return int(match.groups()[0])
        return sorted(frames, key=get_order)
    
    frames = orderFileName(glob.glob(folder+'/*.tif'))
    first = Image.open(frames[0]).convert('LA')
    figsize = np.shape(first)
    figd = figsize[1]
    
    t_data = np.zeros(len(frames[::skipFile]))
    d_data = np.zeros(len(frames[::skipFile]))
    for i, iframe in enumerate(frames[::skipFile]):
        im = Image.open(iframe).convert('LA')
        d, x1, y1, x2, y2 = extractRadius(im)
        t_data[i] = i*1000/(fps/skipFile)
        d_data[i] = d/figd*dPlate
        if np.mod(i, 10) == 0:
            print('Current processing: '+str(i*skipFile)+'/'+str(len(frames)), end='\r')
            im = im.convert('RGB')
            draw = ImageDraw.Draw(im) 
            draw.line((y1, x1, y2, x2), fill=128)
            im.save(folder + '/' + 'centerPoint_' + str(i) + '.png', "png")
    return t_data, d_data

def postProcessDiameter(t0, d0, t_range, window_length=101, polyorder=2, surten= 47.0e-3, dPlate=6):
    nmin = np.argmin(d0)
    # loc = np.where((t0 > t_range[0]) & (t0 < min(t_range[1], nmin)))
    loc = np.where((t0 > t_range[0]) & (t0 < t_range[1]))
    x = t0[loc]
    y = d0[loc]
    delta = t0[1] - t0[0]
    window_length = int(len(x) // 10 * 2 + 1)
    print('Window length: ', window_length)
    smy = savgol_filter(y, window_length, polyorder, delta=delta, mode='nearest')
    dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=delta, mode='nearest')
    hk = 2 * np.log(dPlate / smy[:])
    # hk = 2 * np.log(6 / smy[:])
    etap = - surten / dy[:]
    strt = - 2 * dy[:] / smy[:]
    data = np.array([x, y, dy, strt, hk, etap]).T
    return x, y, data[(window_length//2):(-window_length//2), :]
    # return data