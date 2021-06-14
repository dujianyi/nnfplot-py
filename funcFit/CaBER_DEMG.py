import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def CaBER_DEMG(params, t, manOut=False, rtol=1e-13, atol=1e-13):
    # all with SI units
    # infinite extensibility
    # Adopt the form of Larson with finite extensibility derived from K-H form
    # Separated form of S and lambda
    # L == -1: infinite extensibility
    # adapted from the RP form
    
    def dydt_CaBER_DEMG(t, y, L, tau_d, tau_s):
        if (y[2] > L) and (L > 0):
            return [0, 0, 0]
        else:
            # [R, Srr, Szz, lmd]
            Srr = y[0]
            Szz = y[1]
            lmd = y[2]

            # front factor of f'/f=Cf*lmd_dot, where f is the finite extensibility
            if L==-1:
                f = 1
                Cf = 0
            else:
                f = ((3*(L**2) - (lmd**2))/(3*(L**2) - 1))*(((L**2) - 1)/((L**2) - (lmd**2)))
                Cf = 4*lmd*(L**2)/(3*(L**2)-(lmd**2))/((L**2)-(lmd**2))

            # front factor in epsilon_dot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
            sr_Srr = -2/(Szz-Srr)
            sr_Szz =  2/(Szz-Srr)
            sr_lmd =  4/lmd + 2*Cf

            # Three equations: Srr_dot, Szz_dot, lmd_dot
            relaxTerm = 1/(lmd**2)*(1/tau_d) # constant of disengagement term, check Larson
            
            # front factor in (k:S): k:S = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
            kS_Srr = (Szz-Srr)*sr_Srr
            kS_Szz = (Szz-Srr)*sr_Szz
            kS_lmd = (Szz-Srr)*sr_lmd

            iA = np.linalg.inv([[1+Srr*(   sr_Srr + 2*kS_Srr),   Srr*(   sr_Szz + 2*kS_Szz),   Srr*(   sr_lmd + 2*kS_lmd)],
                                [  Szz*(-2*sr_Srr + 2*kS_Srr), 1+Szz*(-2*sr_Szz + 2*kS_Szz),   Szz*(-2*sr_lmd + 2*kS_lmd)], 
                                [ -lmd*(kS_Srr),                -lmd*(kS_Szz),               1-lmd*(kS_lmd)]])
            iB = [[-relaxTerm*(Srr-1/3)],
                  [-relaxTerm*(Szz-1/3)],
                  [-f/tau_s*(lmd-1)]]
            dy = np.matmul(iA, iB).flatten()
        return dy

    G = params['G']
    tau_d = params['tau_d']
    tau_s = params['tau_s']
    ep0 = params['ep0'] # offset of the strain (initial strain, for fitting of eta vs strain only)
    gm = params['gm']
    R0 = params['R0']
    L = params['L']


    c0 = gm/(3*G*R0)
    if L==-1:
        f = lambda lmd: 1
    else:
        f = lambda lmd: ((3*(L**2) - (lmd**2))/(3*(L**2) - 1))*(((L**2) - 1)/((L**2) - (lmd**2)))

    if (c0 < 1):
        Srr0 = 1/3*(1 -   c0)
        Szz0 = 1/3*(1 + 2*c0)
    else:
        ep0 = fsolve(lambda lmd: f(lmd)*lmd**2-c0, [1])
        Srr0 = 0
        Szz0 = 1

    if type(t) == np.ndarray:
        sol = solve_ivp(dydt_CaBER_DEMG, (0, 1.2*(t.max()-t.min())), [Srr0, Szz0, ep0], t_eval=t-t.min(), args=(L, tau_d, tau_s),
                        rtol=rtol, atol=atol)
    else:
        sol = solve_ivp(dydt_CaBER_DEMG, (0, 1.2*t), [Srr0, Szz0, ep0], args=(L, tau_d, tau_s),
                        rtol=rtol, atol=atol)
    Srr = sol.y[0, :]
    Szz = sol.y[1, :]
    lmd = sol.y[2, :]


    if L==-1:
        f = 1
        Cf = 0
    else:
        f = ((3*(L**2) - (lmd**2))/(3*(L**2) - 1))*(((L**2) - 1)/((L**2) - (lmd**2)))
        Cf = 4*lmd*(L**2)/(3*(L**2)-(lmd**2))/((L**2)-(lmd**2))

    dzz_rr = Szz-Srr

    R = gm/(3*G*f*(lmd**2)*dzz_rr)
    dSol = np.array([dydt_CaBER_DEMG(0, [Srr[i], Szz[i], lmd[i]], L, tau_d, tau_s) for i in range(len(Srr))])

    # calculation of strain rate and viscosity

    # front factor in epsilon_dot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
    sr_Srr = -2/(Szz-Srr)
    sr_Szz =  2/(Szz-Srr)
    sr_lmd =  4/lmd + 2*Cf

    epsilon_dot = sr_Srr*dSol[:, 0] + sr_Szz*dSol[:, 1] + sr_lmd*dSol[:, 2]
    dR = epsilon_dot*R/(-2)
    extVis = gm/(-2*dR)

    # dR21 = dR[:, 1] - dR[:, 0]
    # sr = 2*dR21/dzz_rr
    # st = 2*np.log(R0/R) + ep0
    # vis = gm/R/sr
    # if st_in is not None:
    #     try:
    #         findasd = np.argwhere(st[1:]<=st[:-1])
    #         st = st[:(findasd[0]+1)]
    #         vis = vis[:(findasd[0]+1)]
    #     except:
    #         pass
    #     f_vis = interp1d(st, vis, kind='linear', fill_value='extrapolate')
    #     st = st_in
    #     vis = f_vis(st)

    if manOut==False:
        return R
    return sol.t, Srr, Szz, lmd, R, epsilon_dot, extVis

