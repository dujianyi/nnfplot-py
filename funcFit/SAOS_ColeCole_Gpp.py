import numpy as np

def SAOS_ColeCole_Gpp(params, w):
    Jg = params['Jg']
    eta0 = params['eta0']
    Jr = params['Jr']
    tau0 = params['tau0']
    a = params['a']
    
    J = Jg + 1/(1j*eta0*w) + Jr/((1+1j*tau0*w)**a)
    return np.imag(1/J)