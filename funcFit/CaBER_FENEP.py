import numpy as np

from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def CaBER_FENEP(params, t, nargout):
    # D - mm
    # t - ms
    # others in SI units

    def dydt_CaBER_FENEP(t, y, b, lamb, G0, sigma, shearS, D0):
        # funcFit: FENE-P model, dimensionless
        # b - FENE factor
        # lamb - relaxation time
        # G0 - linear elastic modulus
        # sigma - surface tension
        # shearS - solvent viscosity

        Azz = y[0]
        Arr = y[1]
        D = y[2]
        Z = 1/(1-(Azz+2*Arr)/b)
        epsdot_ast = (2*sigma/(D*D0/1000) - G0*Z*(Azz - Arr))/(3*shearS)*lamb # dimensionless strain rate 
        return np.array([2*epsdot_ast*Azz - (Z*Azz-1), 
                         -epsdot_ast*Arr  - (Z*Arr-1),
                         -epsdot_ast*D/(2)])

    b = params['b']
    lamb = params['lamb']
    G0 = params['G0']
    sigma = params['sigma']
    shearS = params['shearS']
    t0 = params['t0']
    D0 = params['D0']
    
    tspan = (0, 1*(max(t) - t0)/1000./lamb)
    sol = solve_ivp(dydt_CaBER_FENEP, tspan, [np.log(7.7/2.), 1, 1], args=(b, lamb, G0, sigma, shearS, D0), 
                    rtol=1e-4, atol=1e-4)
    f_Azz = interp1d(sol.t, sol.y[0, :], kind='linear', fill_value='extrapolate')
    f_Arr = interp1d(sol.t, sol.y[1, :], kind='linear', fill_value='extrapolate')    
    f_D = interp1d(sol.t, sol.y[2, :], kind='linear')  
    try:
        D = f_D((t-t0)/1000./lamb)*D0
    except:
        print('D')
        return np.ones(len(t))*1e6
    
    try:
        Azz = f_Azz((t-t0)/1000./lamb)
    except:
        print('Azz')
        
    try:
        Arr = f_Arr((t-t0)/1000./lamb)
    except: 
        print('Arr')

    Z = 1/(1 - (Azz + Arr)/b)
    epsdot = (2*sigma/(D/1000) - G0*Z*(Azz - Arr))/(3*shearS)
    exvis = 2*sigma/(D/1000)/(epsdot);

    if nargout == 1:
        return D
    else:
        return D, exvis, epsdot