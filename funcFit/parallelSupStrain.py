
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