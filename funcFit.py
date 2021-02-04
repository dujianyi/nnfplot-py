# save all fitting functions 
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# Processing functions
def refConeEta_DHR(sr, R, T):
    if (T == 'max'):
        Tr = 200e-3
    elif (T == 'min'):
        Tr = 5e-9
    else:
        Tr = T;
    return 3*Tr/(2*np.pi*(R**3)*sr)

def SAOS_FMM_Gp(params, w):
    V = params['V'];
    G = params['G'];
    a = params['a'];
    b = params['b'];
    Gp = (((G*(w**b))**2)*(V*(w**a)*np.cos(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.cos(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    return Gp

def SAOS_FMM_Gpp(params, w):
    V = params['V'];
    G = params['G'];
    a = params['a'];
    b = params['b'];
    Gpp = (((G*(w**b))**2)*(V*(w**a)*np.sin(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.sin(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    return Gpp

def SAOS_Maxwell_Gp(params, w):
    G0 = params['G0']
    tau0 = params['tau0']
    try:
        n = params['n']    # reptation 
    except:
        n = 1
    Gp = 0
    for i in range(1, 2*n+1, 2):
        Gp = Gp + G0/(i**2)*((w*tau0/i**2)**2)/(1+(w*tau0/i**2)**2)
    return Gp

def SAOS_Maxwell_Gpp(params, w):
    G0 = params['G0']
    tau0 = params['tau0']
    try:
        n = params['n']    # reptation 
    except:
        n = 1
    Gpp = 0
    for i in range(1, 2*n+1, 2):
        Gpp = Gpp + G0/(i**2)*((w*tau0/i**2))/(1+(w*tau0/i**2)**2)
    return Gpp

def SAOS_ColeCole_Gp(params, w):
    Jg = params['Jg']
    eta0 = params['eta0']
    Jr = params['Jr']
    tau0 = params['tau0']
    a = params['a']
    
    J = Jg + 1/(1j*eta0*w) + Jr/((1+1j*tau0*w)**a)
    return np.real(1/J)

def SAOS_ColeCole_Gpp(params, w):
    Jg = params['Jg']
    eta0 = params['eta0']
    Jr = params['Jr']
    tau0 = params['tau0']
    a = params['a']
    
    J = Jg + 1/(1j*eta0*w) + Jr/((1+1j*tau0*w)**a)
    return np.imag(1/J)

def SAOS_FKM_Gp(params, w):
    V = params['V']
    G = params['G']
    a = params['a']
    b = params['b']
    G0 = params['G0']
    # Gp = (((G*(w**b))**2)*(V*(w**a)*np.cos(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.cos(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2)) + G0
    Gp = SAOS_FMM_Gp(params, w) + G0
    return Gp

def SAOS_FKM_Gpp(params, w):
    V = params['V']
    G = params['G']
    a = params['a']
    b = params['b']
    # Gpp = (((G*(w**b))**2)*(V*(w**a)*np.sin(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.sin(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    Gpp = SAOS_FMM_Gpp(params, w)
    return Gpp

def HB(params, sr):
    gamma0 = params['gamma0'];
    b = params['b'];
    n = params['n'];
    stress = gamma0 + b*(sr**n)
    return stress

def parallelSupStrain(params, t):
    # for DHR-3
    gm0 = params['gm0']
    b = params['b']
    w = params['w']
    if 'order' in params:
        order = params['order'].value
    else:
        order = 1
    res = gm0*t + b
    for i in range(order):
        res = res + params['gmosc'+str(i)]*np.sin((2*i+1)*w*t + params['ph'+str(i)])
    return res

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

def testFun(params, x):
    a = params['a']
    b = params['b']
    c = params['c']
    return a*(x**2) + b*x + c