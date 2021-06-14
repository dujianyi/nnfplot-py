# functions for Birefringence
from nnfPlot import *
from PIL import Image, ImageStat, ImageOps 
import glob
import os
from scipy import interpolate, ndimage, signal, interpolate
import re
from datetime import datetime

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
    ## plot uncorrected intensity from changing the angle of polarizer and analyzer
    images = sorted(glob.glob(folder+pattern))
    meanMax = meanGreen(Image.open(images[0]))
    meanMin = meanGreen(Image.open(images[-1]))
    xe = np.array([float(os.path.splitext(os.path.basename(i))[0]) for i in images])
    ye = [(meanGreen(Image.open(images[i]))-meanMin)/(meanMax-meanMin) for i in range(0, len(images))]
    plt.plot(xe, ye, 'o')
    x, y = dataOp.refLine(lambda x: np.cos(x*np.pi/180)**2, xrange=[0, 90])
    plt.plot(x, y)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$I/I_0$')
    return xe, ye

def calibrateNoRetard(xe, ye):
    ## calibration of intensity with theoretical cosine results
    x0, y0 = dataOp.refLine(lambda x: np.cos(x*np.pi/180)**2, xin=xe)
    plt.plot(ye, y0, 'o')
    plt.xlabel(r'$I/I_0(imagePixel)$')
    plt.ylabel(r'$I/I_0$')
    xr, yr = dataOp.refLine(lambda x: x, xrange = [0, 1])
    plt.plot(xr, yr, '--')

    f = np.poly1d(np.polyfit(ye, y0, 3))
    xi = np.arange(0, 1, 0.01)
    yi = f(xi)
    plt.plot(xi, yi, '-')
    return f

def calIntensity(image, f, **kwargs):
    ## calculate Intensity with precalibrated function f
    folder = os.path.dirname(image)
    meanMax = meanGreen(Image.open(folder+'/brightest.tif'))
    meanMin = meanGreen(Image.open(folder+'/darkest.tif'))
    return f((meanGreen(Image.open(image), **kwargs)-meanMin)/(meanMax-meanMin))

def plotRetarder(folder, f, wlRetarder, pattern='/*[0-9].tif', wl=525, thetaCorr=0):
    ## retarder: 0-45, brightest, darkest
    images = sorted(glob.glob(folder+pattern))
    ye = np.array([calIntensity(image, f) for image in images])
    xe = np.array([float(os.path.splitext(os.path.basename(i))[0]) for i in images])
    plt.plot(xe, ye, 'o')
    
    delta = 2*np.pi*wlRetarder/wl
    I0 = np.sin(delta/2)**2
    x, y = dataOp.refLine(lambda x: I0*np.sin(2*(x*np.pi/180-np.pi/4+thetaCorr))**2, xrange=[0, 90])
    plt.plot(x, y, '-')
    plt.xlabel(r'$\phi-45^\circ$')
    plt.ylabel(r'$I/I_0$')
    return xe, ye

def calRetarder(xe, ye, wl=525, thetaCorr=0):
    ## calculation of retardation (delta_n*thickness) from fitting a cosine wave
    params = dataOp.Parameters()
    params.add('delta' , value = 0.5,  min = 0,    max = np.pi,   vary = True)
    fun = lambda params, t: (np.sin(params['delta']/2)**2)*np.sin(2*(t*np.pi/180-np.pi/4+thetaCorr))**2

    minner = dataOp.Minimizer(lambda params: fun(params, xe) - ye, params)
    res = minner.minimize()
    # res = params
    
    x, y = dataOp.refLine(lambda x: fun(res.params, x), xrange=[0, 90])
    plt.plot(x, y, '-')
    finres = res.params['delta'].value*wl/(2*np.pi)
    return finres

def logCalc(img, sigma, fsize):
    # function for edgeFinding1D
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


def edgeFinding1D(frame, fig, fsize=4, thresh=1, imgOrient=0):
    
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
    if fig:
        plt.figure()
        plt.plot(diffRes)
    return np.argmin(diffRes), np.argmax(diffRes)

def calScaleBar(folder):
    imgs = glob.glob(folder+'/*.tif')
    tifPattern = '([0-9]*)d([0-9]*)mm'
    dreal = []
    d = []
    for imagei in imgs:
        reMatch = re.search(tifPattern, os.path.splitext(os.path.basename(imagei))[0])
        dreal.append(float(reMatch.group(1)+'.'+reMatch.group(2)))
        d.append(calDiameter(imagei))
    plt.scatter(d, dreal)
    fcoeff = np.polyfit(d, dreal, 1)
    f = np.poly1d(fcoeff)
    xi = np.arange(0, np.max(d), 10)
    yi = f(xi)
    plt.plot(xi, yi, '-')
    return fcoeff[0]

def calDiameter(image, **kwargs):
    pic = Image.open(image)
    y1, y2 = edgeFinding1D(pic, 0, **kwargs)
    return np.abs(y1-y2)

########## IL fibers ##########
def processFiberIandD(images, f, loc='mid', davg=None, fig=False, scaleBar=1, **kwargs):  # scalebar, um/px
    # for IL fibers only, raw data with scaleBar
    # 'mid': return I(1*n) and d(1*n) (dimensional)
    # 'radial': return I(n*R), dd0(n*R)=d/davg (dimensionless)
    if loc == 'mid':
        folder = os.path.dirname(images[0])
        I = np.zeros(len(images))
        d = np.zeros(len(images))
        for i, imagei in enumerate(images):
            pic = Image.open(imagei)
            y1, y2 = edgeFinding1D(pic, fig, **kwargs)
            if fig:
                plt.figure()
                plt.title(os.path.basename(imagei))
                plt.imshow(pic)
                x = [0, pic.size[1]-1]
                plt.plot([y1, y1], x, 'r')
                plt.plot([y2, y2], x, 'b')
            mid = int((y1+y2)/2)
            I[i] = calIntensity(imagei, f, wdRange=[0, mid-10, pic.size[1], mid+10])
            d[i] = (y2-y1)*scaleBar
        return I, d
    elif loc == 'radial':
        folder = os.path.dirname(images[0])
        I = []
        dd0 = []
        for i, imagei in enumerate(images):
            pic = Image.open(imagei)
            y1, y2 = edgeFinding1D(pic, fig, **kwargs)
            mid = int((y1+y2)/2)
            Inow = []
            dd0now = []
            for j in range(0, int(davg/scaleBar/2)):
                Inow.append((calIntensity(imagei, f, wdRange=[0, mid+j, pic.size[1], mid+j+1]) +
                             calIntensity(imagei, f, wdRange=[0, mid-j, pic.size[1], mid-j+1]))/2)
                dd0now.append(2*j*scaleBar/davg)
            I.append(Inow)
            dd0.append(dd0now)
        return I, dd0

def processFiberFolder(folder, patterns, f, scaleBar=1, wl=525, fsize=10, thresh=0.001):  # scalebar, um/px
    ## process a folder of tif files with brightest and darkest in manual mode, where the names are well formatted
    # example: processFiberFolder('20210209/constantDR/', names, f, scaleBar=1, wl=525, fsize=10, thresh=0.001)
    y = np.zeros(len(patterns))
    dm = np.zeros(len(patterns))
    yerr = np.zeros(len(patterns))
    dmerr = np.zeros(len(patterns))
    df = DataFrame()
    for i, nowPattern in enumerate(patterns):
        if (isinstance(nowPattern, str)):
            print('Processing '+nowPattern)
            images = glob.glob(folder+'/'+nowPattern)
        else:
            print('Processing automatic macro '+str(i+1)+'/'+str(len(patterns)))
            images = nowPattern
        I, d = processFiberIandD(images, f, fig=False, scaleBar=scaleBar, fsize=fsize, thresh=thresh)
        y[i] = np.mean(I[:])
        yerr[i] = np.std(I[:])
        dm[i] = np.mean(d[:])
        dmerr[i] = np.std(d[:])
    df['I/I0'] = y
    df['I/I0 error'] = yerr
    df['Diameter (um)'] = dm
    df['Diameter error (um)'] = dmerr
    df['Retardation (nm)'] = np.arcsin(np.sqrt(y))*wl/np.pi # calculation of first order
    df['Retardation error (nm)'] = wl/np.pi/np.sqrt(1-y)/2/np.sqrt(y)*yerr
    df['Birefringence'] = df['Retardation (nm)']/df['Diameter (um)']/1000.
    df['Birefringence error'] = np.sqrt((df['Retardation error (nm)']/df['Diameter (um)']/1000.)**2 + (df['Diameter error (um)']*df['Birefringence']/df['Diameter (um)'])**2)
    df['Name'] = patterns
    return df

def processFiberFolderAuto(folder, matname, f, filHead, filTail, scaleBar=1, wl=525, fsize=10, thresh=0.001):
    ## to process an automatic capturing sequence that is synced with the arduino with a changing godet speed
    ## folder structure: date_folder/folder_seq/*.tif, date_folder/folder_seq/motor.txt, date_folder/background/brightest_001.tif
    
    # read tif
    # example: processFiberFolderAuto('20210210/', 'e5_0d005_23Ga', f, filHead = 0.4, filTail = 0.9, scaleBar=0.00165383*1000, wl=525, fsize=10, thresh=0.001)
    imgPattern = matname+'_([0-9]*_[0-9]*)_*' # extract date+time only
    FMT = '%Y%m%d_%H%M%S'

    imgs = glob.glob(folder+matname+'/'+matname+'*.tif')
    imgsTime = [re.search(imgPattern, os.path.splitext(os.path.basename(imgnow))[0]).group(1) for imgnow in imgs] # string
    firstTime = datetime.strptime(imgsTime[0], FMT)
    imgsTimeInSeconds = np.array([(datetime.strptime(imgTime, FMT) - firstTime).total_seconds() for imgTime in imgsTime]) # unit: second
    argImgsTimeInSeconds = np.argsort(imgsTimeInSeconds)

    imgs = [imgs[i] for i in argImgsTimeInSeconds]
    imgsTimeInSeconds = imgsTimeInSeconds[argImgsTimeInSeconds] - min(imgsTimeInSeconds)

    # read motor
    txtfile = open(folder+matname+'/motor.txt', "r")
    motors = list(filter(None, txtfile.read().split('\n')))

    txtPattern = 'Time interval set to: ([0-9]*) micros, current time milis ([0-9]*)' # extract date+time only
    stepTimeInMS = [int(re.search(txtPattern, txtnow).group(1)) for txtnow in motors] # string
    txtTimeInSeconds = [(int(re.search(txtPattern, txtnow).group(2)) - int(re.search(txtPattern, motors[0]).group(2)))/1000 for txtnow in motors]
    txtTimeInSeconds.append(imgsTimeInSeconds[-1])
    # process
    imgToProcess = []
    for i in range(len(stepTimeInMS)):
        imgwhere = np.where((imgsTimeInSeconds > txtTimeInSeconds[i]) & (imgsTimeInSeconds < txtTimeInSeconds[i+1]))[0]
        n = len(imgwhere)
        nimgwhere = range(int(filHead*n), int(filTail*n))
        imgsNow = [imgs[imgwhere[i]] for i in nimgwhere]
        # print(stepTimeInMS[i], imgsNow)
        imgToProcess.append(imgsNow)
    df = processFiberFolder('', imgToProcess, f, scaleBar=scaleBar, wl=wl, fsize=fsize, thresh=thresh)
    return stepTimeInMS, df, imgToProcess

def postProcessSpatial_processFiberFolderAuto(df, f, wl=525, fsize=10, thresh=0.001):
    # postProcess to calculate the radial distribution of the birefringence
    dfspt = DataFrame()
    
    for i, imagesi in enumerate(df['Name']): # each draw ratio
        print('Processing row '+str(i+1)+'/'+str(len(df['Name'])))
        dfsptnow = DataFrame()
        davg = df['Diameter (um)'][i]
        I, dd0 = processFiberIandD(imagesi, f, loc='radial', fig=False, scaleBar=0.00165383*1000, fsize=fsize, thresh=thresh, davg=davg)
        for j in range(len(I)):  # each image at constant draw ratio
            dfsptnowSingleImage = DataFrame()
            dfsptnowSingleImage['I/I0'] = I[j]
            dfsptnowSingleImage['D/D0'] = dd0[j]
            dfsptnowSingleImage['Thickness (um)'] = davg*np.sqrt(1-np.array(dd0[j])**2)
            dfsptnowSingleImage['Retardation (nm)'] = np.arcsin(np.sqrt(dfsptnowSingleImage['I/I0']))*wl/np.pi # calculation of first order
            dfsptnowSingleImage['Birefringence'] = dfsptnowSingleImage['Retardation (nm)']/dfsptnowSingleImage['Thickness (um)']/1000.
            dfsptnowSingleImage['Name'] = imagesi[j]
            dfsptnow = dfsptnow.append(dfsptnowSingleImage)
        dfsptnow['Strain'] = df['Strain'][i]
        
        dfspt = dfspt.append(dfsptnow)
    return dfspt
