def SAOS_FKM_Gp(params, w):
    V = params['V']
    G = params['G']
    a = params['a']
    b = params['b']
    G0 = params['G0']
    # Gp = (((G*(w**b))**2)*(V*(w**a)*np.cos(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.cos(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2)) + G0
    Gp = SAOS_FMM_Gp(params, w) + G0
    return Gp