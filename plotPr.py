"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
plotPr.py = plot Prandtl number of most first-order models
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np
import matplotlib.pyplot as plt


def WageningenU(Ri):

    Pr = np.zeros_like(Ri)
    for i in range(len(Ri)):

        if Ri[i] > 0 :
            if Ri[i] > 1. : # because \psi --> infinity (cut-off)
                Pr[i] = 1.
            else:
                psi = rootBisect(psiRi_Stable, Ri[i], 0., 1e7)
                Pr[i] = phiStable(psi, 0.8, 7.5)/phiStable(psi, 0.8, 5.0)
        else:
            if Ri[i] < -1e7 : # As if Ri = -infinity (cut-off)
                Pr[i] = 114.72/11397.
            else:
                psi = rootBisect(psiRi_UnStable, Ri[i], -1e7, 0.)
                Pr[i] = phiUnStable(psi, 15., -0.5)/phiUnStable(psi, 20., -0.25)

    return Pr


def psiRi_Stable(psi, Ri):
    return psi*phiStable(psi, 0.8, 7.5) \
             - Ri*phiStable(psi, 0.8, 5.0)**2


def phiStable(psi, a, b):
    return 1. + b*psi*(1.+b*psi/a)**(a-1.)


def psiRi_UnStable(psi, Ri):
    return psi*phiUnStable(psi, 15., -0.5) \
             - Ri*phiUnStable(psi, 20., -0.25)**2


def phiUnStable(psi, g, p):
    return (1. - g*psi)**p


def rootBisect(func, arg, a0, b0, nmax=10000, e=1e-5):
    """
    Find a root of function f by bisection method
    between a and b to tolerance e.
    with Maximum nmax iterations.
    Returns:
    the root and the number of iterations used
    """

    a = a0
    b = b0
    for it in range(nmax):
        c = 0.5*(a+b)
        if func(a,arg)*func(c,arg)<0 :
            b = c
        else:
            a = c

        if abs(func(c,arg)) < abs(e):
            break

    else:
        # Execute if loop ends normally without "break"
        raise Exception("No root found between ", a0," and ", b0,
                        " when Ri =", arg)

    return c


def JMA(Ri):

    Pr = np.zeros_like(Ri)
    Sm = 1.4364382111648997 \
                        *(0.2748189177673626-Ri) \
                        *(0.32487817068599273-Ri) \
                        /(1.-Ri)/(0.3161958453336602 - Ri)
    Sh = 1.9777386666666668*(0.2748189177673626-Ri)/(1.-Ri)
    Pr[ Ri<0.275] = np.divide(Sm,Sh)[ Ri<0.275]
    Pr[ Ri>=0.275 ] = 1.

    return Pr



def plotPr(Ri, PrFunc, names, style='-'):

    font = {"size" : 14}
    plt.rc("font", **font)
    plt.rc('legend', fontsize=11)
    plt.figure(0)

    for i in range(len(PrFunc)) :
        plt.loglog(Ri, PrFunc[i](Ri), style, label=names[i])

    plt.figure(0)
    plt.xlabel(r'Richardson number ($Ri$)')
    plt.ylabel(r'Prandtl nuber ($Pr_t$)')
    plt.ylim(0.6,10)
    plt.savefig('./Result/Pr.png')
    plt.legend()
    plt.tight_layout()
    #plt.grid()
    plt.grid(True, which="both", ls=":", color='0.75')
    plt.show()


def main():

    ECMWF = lambda r : np.divide(1 + 15*r*np.sqrt(1 + 5*r), 1 + 10*r/np.sqrt(1+5*r))
    MetOffice = lambda r : np.ones_like(r)
    NOAA_NCEP = lambda r : 1.5 + 3.08*r
    Pr085 = lambda r : 0.85*np.ones_like(r)
    Surface = lambda r : (1 + 7.8*r)/(1 + 4.8*r)

    # - - - - -

    Ri = np.power(10, np.arange(-4,1, 0.01))

    name_list = ['Pr=0.85','MetOffice/Pr=1']
    model_list = name_list.copy()
    model_list[0] = Pr085
    model_list[1] = MetOffice
    plotPr(Ri, model_list, name_list, style=':')

    name_list = ['ECMWF','NOAA-NCEP','JMA','WageningenU']
    model_list = name_list.copy()
    model_list[0] = ECMWF
    model_list[1] = NOAA_NCEP
    model_list[2] = JMA
    model_list[3] = WageningenU
    plotPr(Ri, model_list, name_list, style='-')

    name_list = ['Surface BC']
    model_list = name_list.copy()
    model_list[0] = Surface
    plotPr(Ri, model_list, name_list, style='--')


main()


