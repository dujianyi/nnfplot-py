import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


def CaBER_RoliePoly(params, t, st_in=None):
    # all with SI units
    # infinite extensibility
    
    def dydt_CaBER_RoliePoly(t, y, tau_d, tau_s, b, dl):
        if y[1] > 1e6:
            return [0, 0]
        else:
            crr = y[0]
            czz = y[1]
            kb = 2*(1-np.sqrt(3/(2*crr + czz)))/tau_s
            dl_in = ((2*crr + czz)/3)**(dl)
            frr = -1/tau_d*(crr-1) - kb*(crr + b*dl_in*(crr-1))
            fzz = -1/tau_d*(czz-1) - kb*(czz + b*dl_in*(czz-1))
            iA = np.linalg.inv([[(czz-3*crr)/(czz-crr), 2*crr/(czz-crr)],
                                   [(4*czz)/(czz-crr)    , (-crr-3*czz)/(czz-crr)]])
            dy = np.matmul(iA, [[frr], [fzz]]).flatten()
        return dy

    G = params['G']
    tau_d = params['tau_d']
    tau_s = params['tau_s']
    ep0 = params['ep0']
    gm = params['gm']
    R0 = params['R0']

    b = 0.5
    dl = -0.5
    c0 = gm/G/R0
    crr0 = fsolve(lambda x: x**2*(c0+x)-1, 1)[0]
    czz0 = 1/(crr0**2)
    
    tspan = (0, 1.2*t)
    sol = solve_ivp(dydt_CaBER_RoliePoly, (0, 1.2*t), [crr0, czz0], args=(tau_d, tau_s, b, dl), 
                    rtol=1e-6, atol=1e-6)
    crr = sol.y[0, :]
    czz = sol.y[1, :]
    dzz_rr = czz-crr
    R = gm/G/dzz_rr
    dR = np.array([dydt_CaBER_RoliePoly(0, [crr[i], czz[i]], tau_d, tau_s, b, dl) for i in range(len(crr))])
    dR21 = dR[:, 1] - dR[:, 0]
    sr = 2*dR21/dzz_rr
    st = 2*np.log(R0/R) + ep0
    vis = gm/R/sr
    if st_in is not None:
        try:
            findasd = np.argwhere(st[1:]<=st[:-1])
            st = st[:(findasd[0]+1)]
            vis = vis[:(findasd[0]+1)]
        except:
            pass
        f_vis = interp1d(st, vis, kind='linear', fill_value='extrapolate')
        st = st_in
        vis = f_vis(st)
    return vis, sr, st, R