
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