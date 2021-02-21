
def HB(params, sr):
    gamma0 = params['gamma0'];
    b = params['b'];
    n = params['n'];
    stress = gamma0 + b*(sr**n)
    return stress