# functions for LAOS
from nnfPlot import *
from numba import jit
import re

def LAOSread(filename, strain, sinewave, category):
    rawxls = [fileOp.readXLS(filename, reindex=1, appTag = {'Concentration':category, 'Strain':'{:.3f}'.format(strain[i]*100)+'%'}, sheet_name='Sine Strain - '+str(j), skiprows=[0], header = [0, 1]) for i, j in enumerate(sinewave)]
    LAOSread = pd.concat(rawxls, axis=0)
    LAOSread.rename(columns=lambda x: re.sub('(.+)\s+\(Unnamed:.+\)', r'\1 (-)', x), inplace=True)
    return LAOSread

def LAOS3Dplot(df, fit_on=False, sp_per_cycle=383.5, hOrder=3, maxx=None, maxy=None, maxz=None):
    fig = plt.figure(figsize=(12, 8))
    strain = df['Strain'].unique()
    col = sns.color_palette(n_colors=len(strain))
    ax = fig.add_subplot(111, projection = '3d')
    ax.tick_params(axis='both', which='major', pad=0)
    if (maxx == None):
        maxx = 0
        for (i, ist) in enumerate(strain):
            stDf = df[df['Strain']==ist]
            x = stDf['Strain (-)']
            maxx = max(maxx, max(x))
    if (maxy == None):
        maxy = 0
        for (i, ist) in enumerate(strain):
            stDf = df[df['Strain']==ist]
            y = stDf['Shear rate (1/s)']
            maxy = max(maxy, max(y))
    if (maxz == None): 
        maxz = 0
        for (i, ist) in enumerate(strain):
            stDf = df[df['Strain']==ist]
            z = stDf['Stress (Pa)']
            maxz = max(maxz, max(z))

#         if fit_on:
#             x_new, y_new, z_new = LAOS13harmonics(stDf, 383.5)
#             ax.plot(x_new, y_new, z_new, color=col[i], linewidth=1)
    
    kfac = 1.2
    ax.set_xlim([-maxx*kfac, maxx*kfac])
    ax.set_ylim([-maxy*kfac, maxy*kfac])
    ax.set_zlim([-maxz*kfac, maxz*kfac])
        
    for (i, ist) in enumerate(strain):
        stDf = df[df['Strain']==ist]
        x = stDf['Strain (-)']
        y = stDf['Shear rate (1/s)']
        z = stDf['Stress (Pa)']
        zero = np.ones(len(z))
        ax.plot(zero*(-maxy*kfac), y, z, 'o', markevery=1, markerfacecolor='none', c=col[i], markersize=3, alpha=0.2)
        ax.plot(x, zero*(maxx*kfac), z, 'o', markevery=1, markerfacecolor='none', c=col[i], markersize=3, alpha=0.2)
        if fit_on:
            x_new, y_new, z_new = LAOS13harmonics(stDf, sp_per_cycle, hOrder=hOrder)
            ax.plot(zero*(-maxy*kfac), y_new, z_new, color=col[i], linewidth=1, alpha=0.5)
            ax.plot(x_new, zero*(maxx*kfac), z_new, color=col[i], linewidth=1, alpha=0.5)
            
    for (i, ist) in enumerate(strain):
        stDf = df[df['Strain']==ist]
        x = stDf['Strain (-)']
        y = stDf['Shear rate (1/s)']
        z = stDf['Stress (Pa)']
        ax.plot(x, y, z, 'o', markevery=1, c=col[i], markerfacecolor='none', markersize=4)
        
    ax.set_xlabel('Strain (-)')
    ax.set_ylabel('Shear rate (1/s)')
    ax.set_zlabel('Stress (Pa)')
  
    plt.show()
    
def LAOS13harmonics(df, sp_per_cycle, fft=False, w0=1, hOrder=3): # for one strain

    from scipy import fftpack

    T = 2*np.pi/sp_per_cycle
    t = df['Step time (s)'].to_numpy()
    sr = df['Strain (-)'].to_numpy()
    srate = df['Shear rate (1/s)'].to_numpy()
    st = df['Stress (Pa)'].to_numpy()
    N = len(t)
    sr_fft = fftpack.fft(sr)
    srate_fft = fftpack.fft(srate)
    st_fft = fftpack.fft(st)
    freq = fftpack.fftfreq(N) * 2*np.pi/(T)

    stabsort = np.argsort(np.abs(st_fft[:(N//2)]))
    sr_fft_new = np.zeros(N, dtype=complex)
    srate_fft_new = np.zeros(N, dtype=complex)
    st_fft_new = np.zeros(N, dtype=complex)
    
    amax1 = np.round(w0*N/sp_per_cycle).astype(int)

    sr_fft_new[amax1] = sr_fft[amax1]
    sr_fft_new[N-amax1] = sr_fft[N-amax1]
    srate_fft_new[amax1] = srate_fft[amax1]
    srate_fft_new[N-amax1] = srate_fft[N-amax1]
    
    for i in range(0, int((hOrder+1)/2)):
        amax = amax1*(2*i+1)
        st_fft_new[amax] = st_fft[amax]
        st_fft_new[N-amax] = st_fft[N-amax]


#     st_fft_new[amax1] = st_fft[amax1]
#     st_fft_new[amax2] = st_fft[amax2]
#     st_fft_new[N-amax1] = st_fft[N-amax1]
#     st_fft_new[N-amax2] = st_fft[N-amax2]

    sr_new = fftpack.ifft(sr_fft_new)  
    srate_new = fftpack.ifft(srate_fft_new)
    st_new = fftpack.ifft(st_fft_new)

#     plt.subplot(231)
#     plt.plot(t, sr, 'o', t, sr_new, '-')
#     plt.subplot(232)
#     plt.plot(t, srate, 'o', t, srate_new, '-')
#     plt.subplot(233)
#     plt.plot(t, st, 'o', t, st_new, '-')
#     plt.subplot(234)
#     plt.semilogy(freq, np.abs(sr_fft), '.')
#     # plt.xlim([-10, 10])
#     plt.subplot(235)
#     plt.semilogy(freq, np.abs(srate_fft), '.')
#     # plt.xlim([-10, 10])
#     plt.subplot(236)
#     plt.semilogy(freq, np.abs(st_fft), '.')
#     # plt.xlim([-10, 10])
    if fft:
        return sr_new, srate_new, st_new, sr_fft_new, srate_fft_new, st_fft_new
    else:
        return sr_new, srate_new, st_new

def LAOSlissajous(df, xword='Strain (-)', yword='Stress (Pa)', fit_on=False, sp_per_cycle=383.5, columns=5, hOrder=3):
    grid = sns.FacetGrid(data=df, col='Strain', hue='Strain', height=4.3, col_wrap=columns,
                    palette='tab20c', despine=False, sharex=False, sharey=False)
    grid.map_dataframe(sns.scatterplot, xword, yword)
    if fit_on:
        axes = grid.fig.axes
        strain = df['Strain'].unique()
        for i, istrain in enumerate(strain):
            dfnow = df[df['Strain'] == istrain]
            t_new = dfnow['Step time (s)']
            x_new, y_new, z_new = LAOS13harmonics(dfnow, sp_per_cycle, hOrder=hOrder)
            to_plot = {'Step time (s)':    t_new, 
                       'Strain (-)':       x_new, 
                       'Shear rate (1/s)': y_new, 
                       'Stress (Pa)':      z_new}
            axes[i].plot(to_plot[xword], to_plot[yword], 'b-')
    return grid

def LAOSI31gamma2(df, sp_per_cycle=383.5, nlOrder=3, relVal=False, **kwargs):
    strain = df['Strain'].unique()
    f_st = lambda x: float(x.strip('%'))/100.
    st = [f_st(i) for i in strain]
    q = np.zeros(len(strain))
    for i, istrain in enumerate(strain):
        stDf = df[df['Strain']==istrain]
        sr_new, srate_new, st_new, sr_fft_new, srate_fft_new, st_fft_new = LAOS13harmonics(stDf, sp_per_cycle, hOrder=nlOrder, fft=True)
        N = len(stDf)
        asort = np.argsort(np.abs(st_fft_new[:N//2]))
        Ii = lambda od: np.abs(st_fft_new[asort[-int((od+1)/2)]])/(len(st_new)/2)
        if relVal:
            q[i] = Ii(nlOrder)/Ii(1)/st[i]**(nlOrder-1)
            # q[i] = (np.abs(st_fft_new[asort[-1]])/np.abs(st_fft_new[asort[-int((nlOrder+1)/2)]]))**(1)/(np.abs(sr_fft_new[asort[-1]])/np.abs(sr_fft_new[asort[-int((nlOrder+1)/2)]]))
        else:
            q[i] = Ii(nlOrder)/st[i]**(nlOrder)

    plt.loglog(st, q, **kwargs)
    plt.xlabel(r'$\gamma_0$')
    if relVal:
        plt.ylabel(r'$Q=(I_{{{order}}}/I_1)/\gamma_0^{{{power}}}$'.format(order=str(nlOrder), power=str(nlOrder-1)))
    else:
        plt.ylabel(r'$I_{{{order}}}/\gamma_0^{{{power}}}$ (Pa)'.format(order=str(nlOrder), power=str(nlOrder)))
    
def LAOSGpGpp(df, sp_per_cycle=383.5, fig='tan', **kwargs):
    from scipy import fftpack

    strain = df['Strain'].unique()
    f_st = lambda x: float(x.strip('%'))/100.
    st = [f_st(i) for i in strain]
    Gp = np.zeros(len(strain))
    Gpp = np.zeros(len(strain))
    tand = np.zeros(len(strain))
    for i, istrain in enumerate(strain):

        stDf = df[df['Strain']==istrain]
        sr_new, srate_new, st_new, sr_fft_new, srate_fft_new, st_fft_new = LAOS13harmonics(stDf, sp_per_cycle, fft=True)
        N = len(stDf)
        asort = np.argsort(np.abs(st_fft_new[:N//2]))
        a_sr = sr_fft_new[asort[-1]].real
        b_sr = sr_fft_new[asort[-1]].imag
        tand_sr = np.arctan(-a_sr/b_sr)
        a_st = st_fft_new[asort[-1]].real
        b_st = st_fft_new[asort[-1]].imag
        tand_st = np.arctan(-a_st/b_st)
        tand[i] = np.mod(tand_st - tand_sr, np.pi)
        
        Gp[i] = np.abs(st_fft_new[asort[-1]])/np.abs(sr_fft_new[asort[-1]])*np.cos(tand[i])
        Gpp[i] = np.abs(st_fft_new[asort[-1]])/np.abs(sr_fft_new[asort[-1]])*np.sin(tand[i])
        # Gp[i] = np.abs(st_fft_new[asort[-1]])/st[i]*np.cos(tand[i])
        # Gpp[i] = np.abs(st_fft_new[asort[-1]])/st[i]*np.sin(tand[i])
    # return st, Gp, Gpp
    if (fig == 'G'):
        plt.loglog(st, Gp, 'o', color=kwargs['col'])
        plt.loglog(st, Gpp, 'o', color=kwargs['col'], markerfacecolor='none')
        plt.xlabel(r'$\gamma_0$')
        plt.ylabel(r'$G\prime, G{\prime\prime}$ (Pa)')
    elif (fig == 'tan'):
        plt.loglog(st, np.tan(tand), 'o', markerfacecolor='none')
        # plt.loglog(st, tand, 'o', markerfacecolor='none')
        plt.xlabel(r'$\gamma_0$')
        plt.ylabel(r'$\tan \delta $')
    
