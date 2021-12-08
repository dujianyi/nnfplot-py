import numpy as np

def SAOS_FZM_Gpp(params, w):
    V = params['V']
    G = params['G']
    a = params['a']
    b = params['b']
    G0 = params['G0']
    g = params['g']
    # Gpp = (((G*(w**b))**2)*(V*(w**a)*np.sin(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.sin(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    Gpp = (((G*(w**b))**2)*(V*(w**a)*np.sin(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.sin(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2)) + G0*(w**g)*np.sin(np.pi*g/2)
    return Gpp