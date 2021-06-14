import numpy as np
    
def LAOS_KBKZ(m, h, gamma, t, int_period=10, sp_per_cycle=700, **kwargs):
    # characteristic timescale
    tau = kwargs['tau']
    epsi = np.linspace(0, int_period*tau, int_period*sp_per_cycle+1)
    return [np.trapz(m(epsi)*h(gamma(ti)-gamma(ti-epsi))*(gamma(ti)-gamma(ti-epsi)), epsi) for ti in t]
