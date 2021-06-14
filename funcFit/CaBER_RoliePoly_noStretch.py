import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def CaBER_RoliePoly_noStretch(params, t):
    # all with SI units
    # infinite extensibility
    # no chain stretch
    # Larson 11.16 and 11.11(sigma=3GS)

    def dydt_CaBER_RoliePoly_noStretch(t, y, tau_d, b):

        # [R, Srr, Szz, lmd]
        Srr = y[0]
        Szz = y[1]

        # front factor of f'/f=Cf*lmd_dot, where f is the finite extensibility

        # front factor in epsilon_d3ot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
        sr_Srr = -2/(Szz-Srr)
        sr_Szz =  2/(Szz-Srr)
        # front factor in  (k:S): k:S = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
        kS_Srr = -2 # =(Szz-Srr)*sr_Srr
        kS_Szz = 2  # =(Szz-Srr)*sr_Szz

        iA = np.linalg.inv([[1+Srr*(   sr_Srr + 2*kS_Srr) + 2*b*(Srr-1/3)*kS_Srr,   Srr*(   sr_Szz + 2*kS_Szz) + 2*b*(Srr-1/3)*kS_Szz],
                            [  Szz*(-2*sr_Srr + 2*kS_Srr) + 2*b*(Szz-1/3)*kS_Srr, 1+Szz*(-2*sr_Szz + 2*kS_Szz) + 2*b*(Szz-1/3)*kS_Szz]])
        iB = [[-(Srr-1/3)/tau_d],
              [-(Szz-1/3)/tau_d]]
        dy = np.matmul(iA, iB).flatten()
        return dy

    G = params['G']
    tau_d = params['tau_d']
    gm = params['gm']
    R0 = params['R0']

    b = 1
    c0 = gm/(3*G*R0)
    # assume initial stretch of 1
    Srr0 = 1/3*(1 -   c0)
    Szz0 = 1/3*(1 + 2*c0)

    sol = solve_ivp(dydt_CaBER_RoliePoly_noStretch, (0, 1.2*(t.max() - t.min())), [Srr0, Szz0], method="RK23", args=(tau_d, b),
                    rtol=1e-4, atol=1e-4)
    Srr = sol.y[0, :]
    Szz = sol.y[1, :]


    dzz_rr = Szz-Srr

    R = gm/(3*G*dzz_rr)
    dSol = np.array([dydt_CaBER_RoliePoly_noStretch(0, [Srr[i], Szz[i]], tau_d, b) for i in range(len(Srr))])

    # calculation of strain rate and viscosity

    # front factor in epsilon_dot: epsilon_dot = sr_Srr*Srr_dot + sr_Szz*Szz_dot + sr_lmd*lmd_dot 
    sr_Srr = -2/(Szz-Srr)
    sr_Szz =  2/(Szz-Srr)

    epsilon_dot = sr_Srr*dSol[:, 0] + sr_Szz*dSol[:, 1]
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
    return sol.t, Srr, Szz, R, epsilon_dot, extVis