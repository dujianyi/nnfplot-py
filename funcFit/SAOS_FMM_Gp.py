import numpy as np

def SAOS_FMM_Gp(params, w):
    V = params['V'];
    G = params['G'];
    a = params['a'];
    b = params['b'];
    Gp = (((G*(w**b))**2)*(V*(w**a)*np.cos(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.cos(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    return Gp