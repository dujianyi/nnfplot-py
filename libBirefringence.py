# functions for Birefringence
from nnfPlot import *
from PIL import Image, ImageStat, ImageOps 
import glob
import os
from scipy import interpolate, ndimage, signal, interpolate
from skimage import measure

def meanGreen(frame, wdRange=[0, 0, 0, 0]):
    ##  calculate the greeness from a single image
    if all(v == 0 for v in wdRange):
        sh = np.shape(frame)
        selrange = [0, 0, sh[0], sh[1]]
    else: 
        selrange = wdRange
    output = np.array(ImageOps.grayscale(frame))
#     plt.figure()
#     plt.imshow(output[selrange[0]:selrange[2], selrange[1]:selrange[3]])
    return np.mean(output[selrange[0]:selrange[2], selrange[1]:selrange[3]])

def plotNoRetard(folder, pattern='/*.tif'):
    images = sorted(glob.glob(folder+pattern))
    meanMax = meanGreen(Image.open(images[0]))
    meanMin = meanGreen(Image.open(images[-1]))
    xe = np.array([float(os.path.splitext(os.path.basename(i))[0]) for i in images])
    ye = [(meanGreen(Image.open(images[i]))-meanMin)/(meanMax-meanMin) for i in range(0, len(images))]
    plt.plot(xe, ye, 'o')
    x, y = refLine(lambda x: np.cos(x*np.pi/180)**2, xrange=[0, 90])
    plt.plot(x, y)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$I/I_0$')
    return xe, ye

def calibrateNoRetard(xe, ye):
    # calibration
    x0, y0 = refLine(lambda x: np.cos(x*np.pi/180)**2, xin=xe)
    plt.plot(ye, y0, 'o')
    plt.xlabel(r'$I/I_0(imagePixel)$')
    plt.ylabel(r'$I/I_0$')
    xr, yr = refLine(lambda x: x, xrange = [0, 1])
    plt.plot(xr, yr, '--')

    f = np.poly1d(np.polyfit(ye, y0, 3))
    xi = np.arange(0, 1, 0.01)
    yi = f(xi)
    plt.plot(xi, yi, '-')
    return f

def calIntensity(images, folder, f, **kwargs):
    # calculate Intensity with precalibrated function f
    meanMax = meanGreen(Image.open(folder+'/brightest.tif'))
    meanMin = meanGreen(Image.open(folder+'/darkest.tif'))
    if isinstance(images, str):
        return f((meanGreen(Image.open(images), **kwargs)-meanMin)/(meanMax-meanMin))
    else:
        return f([(meanGreen(Image.open(i), **kwargs)-meanMin)/(meanMax-meanMin) for i in images])

def plotRetarder(folder, f, wlRetarder, wl=525, thetaCorr=0):
    # retarder: 0-45, brightest, darkest
    images = sorted(glob.glob(folder+pattern))
    ye = calIntensity(images, folder, f)
    xe = np.array([float(os.path.splitext(os.path.basename(i))[0]) for i in images])
    plt.plot(xe, ye, 'o')
    
    delta = 2*np.pi*wlRetarder/wl
    I0 = (np.sin(2*(np.pi/4))**2)*np.sin(delta/2)**2
    x, y = refLine(lambda x: I0*np.sin(2*(x*np.pi/180-np.pi/4+thetaCorr))**2, xrange=[0, 90])
    plt.plot(x, y, '-')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$I/I_0$')
    return xe, ye

def calRetarder(xe, ye, wl=525, thetaCorr=0):
    # calculation of retardation (delta_n*thickness) from 
    params = Parameters()
    params.add('delta' , value = 0.5,  min = 0,    max = np.pi,   vary = True)
    fun = lambda params, t: (np.sin(params['delta']/2)**2)*np.sin(2*(t*np.pi/180-np.pi/4+thetaCorr))**2

    minner = Minimizer(lambda params: fun(params, xe) - ye, params)
    res = minner.minimize()
    # res = params
    
    x, y = refLine(lambda x: fun(res.params, x), xrange=[0, 90])
    plt.plot(x, y, '-')
    finres = res.params['delta'].value*wl/(2*np.pi)
    return finres

def logCalc(img, sigma, fsize):
    siz   = (sigma-1)/2
    std2  = fsize**2
    x = np.arange(-siz, siz+1, 1)
    xv, yv = np.meshgrid(x, x)
    arg = -(xv**2 + yv**2)/(2*std2)
    h = np.exp(arg)
    sumh = np.sum(h)
    if not sumh == 0:
        h = h/sumh
    h1 = np.multiply(h, (xv**2 + yv**2 - 2*std2)/(std2**2))
    h = h1 - np.sum(h1)/sigma
    h = h - np.sum(h)/np.size(h)
    return ndimage.convolve(img, h, mode='nearest')


########## IL fibers ##########

def edgeFinding1D(frame, fsize=4, thresh=1, imgOrient=0):
    
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
        sumh = np.sum(h)
        if not sumh == 0:
            h = h/sumh
        h1 = np.multiply(h, (xv**2 + yv**2 - 2*std2)/(std2**2))
        h = h1 - np.sum(h1)/sigma
        h = h - np.sum(h)/np.size(h)
        return ndimage.convolve(img, h, mode='nearest')
    
    # imgOrient: 0, vertical fiber; 1, horizontal fiber
    sigma = fsize*6+1
    # img = normalizeGray(frame)
    img = np.array(ImageOps.grayscale(frame))
    img = img/np.max(img)
    # result = ndimage.gaussian_laplace(img, sigma=sigma)
    result = logCalc(img, sigma, fsize)
    result = result.sum(axis=imgOrient)
    
    diffRes = result[1:]-result[:-1]
    plt.figure()
    plt.plot(diffRes)
    return np.argmin(diffRes), np.argmax(diffRes)

def processFiber(folder, f, pattern='/*.tif', loc='mid', fig=True, **kwargs):
    # for IL fibers only
    images = glob.glob(folder+pattern)
    I = np.zeros(len(images))
    d = np.zeros(len(images))
    for i, imagei in enumerate(images):
        pic = Image.open(imagei)
        y1, y2 = edgeFinding1D(pic, **kwargs)
        if fig:
            plt.figure()
            plt.title(os.path.basename(imagei))
            plt.imshow(pic)
            x = [0, pic.size[1]-1]
            plt.plot([y1, y1], x, 'r')
            plt.plot([y2, y2], x, 'b')

        if (loc == 'mid'):
            mid = int((y1+y2)/2)
            I[i] = calIntensity(imagei, folder, f, wdRange=[0, mid-10, pic.size[1], mid+10])
        d[i] = y2-y1
    return I, d