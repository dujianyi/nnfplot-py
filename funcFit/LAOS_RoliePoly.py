import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def LAOS_RoliePoly(params, t, input_sr):
    # all with SI units
    # parameters in p477 Dealy and Larson, 2006
    # input_sr = strain rate(t)
    # shear flow only with arbitrary time-dependent shear rate
    
    def dydt_LAOS_RoliePoly(t, y, sr, tauD, tauS, delta, beta):

        Sxx = y[0]
        Sxy = y[1]
        Syy = y[2]
        lmd = y[3]
        kcolonS = sr(t)*Sxy
        BS = (1/tauD + 2*beta*(lmd-1)/tauS*(lmd**(delta-1)))
        dy = np.array([2*Sxy*sr(t) - 2*kcolonS*Sxx - 1/(lmd**2)*BS*(Sxx-1/3), 
                         Syy*sr(t) - 2*kcolonS*Sxy - 1/(lmd**2)*BS*(Sxy),
                                   - 2*kcolonS*Syy - 1/(lmd**2)*BS*(Syy-1/3),
                       lmd*kcolonS - 1/tauS*(lmd-1) - BS*((lmd**2)-1)/(2*lmd)])
        return dy

    # LVE properties in input_LVE

    # plastic properties
    tauD = params['tauD']
    tauS = params['tauS']
    delta = params['delta']
    beta = params['beta']

    tspan = (0, 1.2*t)
    sol = solve_ivp(dydt_LAOS_RoliePoly, (0, 1.2*t), [1/3, 0, 1/3, 1], t_eval=np.linspace(0, 1.2*t, 1000), args=(input_sr, tauD, tauS, delta, beta))

    t_sol = sol.t[:]
    Sxx_sol = sol.y[0, :]
    Sxy_sol = sol.y[1, :]
    Syy_sol = sol.y[2, :]
    lmd_sol = sol.y[3, :]
    return t_sol, Sxx_sol, Sxy_sol, Syy_sol, lmd_sol