def SAOS_FKM_Gpp(params, w):
    V = params['V']
    G = params['G']
    a = params['a']
    b = params['b']
    # Gpp = (((G*(w**b))**2)*(V*(w**a)*np.sin(np.pi*a/2)) + ((V*(w**a))**2)*(G*(w**b)*np.sin(np.pi*b/2)))/((V*(w**a))**2 + (G*(w**b))**2 + 2*V*G*(w**(a+b))*np.cos(np.pi*(a-b)/2))
    Gpp = SAOS_FMM_Gpp(params, w)
    return Gpp