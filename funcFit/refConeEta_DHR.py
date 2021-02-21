# reference cone-plate shear viscosity

import numpy as np

def refConeEta_DHR(sr, R, T):
    if (T == 'max'):
        Tr = 200e-3
    elif (T == 'min'):
        Tr = 5e-9
    else:
        Tr = T;
    return 3*Tr/(2*np.pi*(R**3)*sr)