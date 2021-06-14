import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def LAOS_IKH(params, t, input_sr, input_LVE):
    # all with SI units
    # parameters in Dimitriou 2014, Soft Matter
    # input_strain = strain(t)
    # input_sr = strain rate(t)
    # sigma_dot = input_LVE(params, t, input_sr, sigma)
    
    def dydt_LAOS_IKH(t, y, sr, f_LVE, mu_p, k1, k2, k3, C, q, m):
        sigma = y[0]
        A = y[1]
        lmd = y[2]
        sigma_eff = sigma - C*A
        if np.abs(sigma_eff) < k3*lmd: # plastic strain rate
            gamma_dot_p = 0
        else:
            gamma_dot_p = np.sign(sigma_eff)*(np.abs(sigma_eff)-k3*lmd)/mu_p

        dy = np.array([f_LVE(t, sr(t)-gamma_dot_p, sigma), 
                       gamma_dot_p - ((q*np.abs(A))**m)*np.sign(A)*np.abs(gamma_dot_p), 
                       k1*(1-lmd) - k2*lmd*np.abs(gamma_dot_p)])
        return dy

    # LVE properties in input_LVE

    # plastic properties
    mu_p = params['mu_p']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    C = params['C']
    q = params['q']
    m = params['m']

    f_LVE = lambda t, sr, sigma: input_LVE(params, t, sr, sigma)

    tspan = (0, 1.2*t)
    sol = solve_ivp(dydt_LAOS_IKH, (0, 1.2*t), [0, 0, 1], args=(input_sr, f_LVE, mu_p, k1, k2, k3, C, q, m))

    t_sol = sol.t[:]
    sigma_sol = sol.y[0, :]
    A_sol = sol.y[1, :]
    lmd_sol = sol.y[2, :]

    return t_sol, sigma_sol, A_sol, lmd_sol