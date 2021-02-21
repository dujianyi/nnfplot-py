import numpy as np
from scipy.optimize import fsolve

def CaBER_FENEP_elastic(params, t, nargout=1):
    # D - mm
    # t - ms
    # others in SI units
    
    b = params['b']
    Ec = params['Ec']
    lamb = params['lamb']
    t0 = params['t0']
    D0 = params['D0']

    tau = (t-t0)/1000/lamb
    f = lambda x, t: (1/(1+Ec*(b+3)) - 1/(1+x*Ec*(b+3))) + 3*np.log((1+x*Ec*(b+3))/(1+Ec*(b+3))) + 4*Ec*(b+3)/(b+2)*(x-1) + (b+3)**2/(b*(b+2))*t
    
    dia = np.zeros(len(tau))
    eps = 1e-14
    triInit = np.logspace(0, -4, 20)
    for (i, itau) in enumerate(tau):
        sol_x = 1e20
        for j in triInit:
            try:
                temp = fsolve(f, j, args=itau)
                if (np.isreal(temp[0])) and (abs(f(temp[0], itau))<eps) and (temp[0]>0):
                    sol_x = temp[0]
                    break
            except:
                pass
        dia[i] = sol_x
    return dia*D0