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