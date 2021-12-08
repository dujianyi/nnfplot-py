# Data processing
from lmfit import Parameters, Minimizer


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
    import numpy as np

    if xin is None:
        x = np.linspace(xrange[0], xrange[1], 100)
        return x, fun(x)
    else:
        return xin, fun(xin)

def fitLine(initParams, fitPack=None, fitRange=None, logFit=False, minFun=None, **kwargs):
    # data is dataFrame, x, y, minFun is list of str/functions to fit correspondingly
    import numpy as np

    def objective(params):
        res = []
        for i, (data, ix, iy, f) in enumerate(fitPack):
            datacopy = data.dropna(subset=[ix, iy])
            if fitRange is not None:
                datacopy = datacopy[(datacopy[ix]>=fitRange[i][0]) & (datacopy[ix]<=fitRange[i][1])]
            if logFit:
                datacopy = datacopy[datacopy[iy]>0]
                res.append(np.log(f(params, datacopy[ix].to_numpy()))-np.log(datacopy[iy].to_numpy()))
            else:
                res.append(f(params, datacopy[ix].to_numpy())-datacopy[iy].to_numpy())
        res = np.concatenate([i for i in res])
        print('Now fitting:', iy, 'Size:', res.shape, end='\r')
        return res

    if minFun is None:
        minner = Minimizer(objective, initParams)
    else:
        # Directly given minFun
        minner = Minimizer(minFun, initParams)

    print('Start fitting...')
    try:
        result = minner.minimize(**kwargs)
        print('Finished fitting...')
        return result
    except TypeError: 
        print('Errors occur in evaluating the function.')
        raise

    #np.sum(result.residual)