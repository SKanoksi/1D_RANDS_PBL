"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
FirstOrderModels.py = some implementations of first-order turbulence model
- ECMWF
- MeteoFrance
- NOAA_NCEP
- JMA
- MetOffice
- WageningenU
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np

from abstractModels import Setup, First_Order_model
from BC import Boundary_Cuxart2006 as defaultBC


class ECMWF(First_Order_model):
    """
    ECMWF-MO first-order turbulence model (Belijaars and Viterbo, 1988)
    (lm0 = lh0 = 150 m)
    (derived from "First_Order_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./150.)
        lm2 = np.multiply(lm,lm)

        return [lm2, lm2]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For stable (Ri>0) ONLY
        if np.any(Ri < 0) :
            print('This model is not for unstable cases yet.')
        temp = np.sqrt(1. + 5.*Ri)
        fm = 1./(1. + 10.*np.divide(Ri, temp))
        fh = 1./(1. + 15.*np.multiply(Ri, temp))

        return [fm, fh]




class MeteoFrance(First_Order_model):
    """
    MeteoFrance first-order turbulence model (Louis et al., 1982)
    (lm0 = 150 m, Lh0 = 450 m)
    derived from "First_Order_model" class
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./150.)
        lh = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./450.)

        return [np.multiply(lm,lm), np.multiply(lh,lh)]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For stable (Ri>0) ONLY
        if np.any(Ri < 0) :
            print('This model is not for unstable cases yet.')
        temp = np.sqrt(1. + 5.*Ri)
        fm = 1./(1. + 10.*np.divide(Ri, temp))
        fh = 1./(1. + 15.*np.multiply(Ri, temp))

        return [fm, fh]




class NOAA_NCEP(First_Order_model):
    """
    NOAA-NCEP first-order turbulence model (Hong and Pan, 1996)
    (lm = 250 m (operational) not 30 m (in the paper))
    derived from "First_Order_model" class
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./250.)
        lm2 = np.multiply(lm,lm)

        return [lm2, lm2]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For stable (Ri>0) ONLY
        if np.any(Ri < 0) :
            print('This model is not for unstable cases yet.')
        fm = np.exp(-8.5*Ri) + 0.15/(Ri + 3.0)
        fh = np.divide(fm, 1.5 + 3.08*Ri)

        return [fm, fh]




class JMA(First_Order_model):
    """
    JMA first-order model (Yamada, 1975)
    (lm0 = lh0 = 50 m)
    (derived from "First_Order_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./50.) # Cuxart 2006
        lm2 = np.multiply(lm,lm)

        return [lm2, lm2]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # Both stable (Ri>0) and unstable (Ri<0) ???
        mask = Ri < 0.2748189177673626  # Critical Ri = around 0.275
        Sm = np.zeros_like(Ri)
        Sh = np.zeros_like(Ri)
        coeff = np.zeros_like(Ri)
        Sm[mask] = 1.4364382111648997*(0.2748189177673626 - Ri[mask]) \
                    *(0.32487817068599273 - Ri[mask])/(1. - Ri[mask]) \
                    /(0.3161958453336602 - Ri[mask])
        Sh[mask] = 1.9777386666666668*(0.2748189177673626 - Ri[mask])/(1. - Ri[mask])
        coeff[mask] = np.sqrt(15.*np.multiply(1. - Ri[mask], Sm[mask]))

        fm = np.multiply(coeff, Sm)
        fh = np.multiply(coeff, Sh)

        return [fm, fh]




class MetOffice(First_Order_model):
    """
    MetOffice first-order turbulence model (Williams, 2002) (Louis, 1974)
    (lm0 = lh0 = 100 m)
    (derived from "First_Order_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = 1./(1./self.BC.k/self.z_secondary[1:-1] + 1./100.) # Louis, 1974
        lm2 = np.multiply(lm,lm)

        return [lm2, lm2]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For both stable (Ri>0) and unstable (Ri<0)
        fm = np.zeros_like(Ri)
        fh = np.zeros_like(Ri)
        for i in range(len(Ri)):
            if Ri[i] > 0 :
                #fm[i] = 1./(1. + 10.*Ri) # Cuxart, 2006
                fm[i] = (1. + 4.7*Ri[i])**-2 # Louis, 1974
                fh[i] = fm[i]
            else:
                z = self.z_secondary[i+1]
                dz = self._dz
                l = 1./(1./self.BC.k/z + 1./100.)
                c = l*l*9.4*((z+dz/z)**(1/3) -1.)**1.5*(z**-0.5)*(dz**-1.5)
                fm[i] = 1. - 9.4*Ri[i]/(1. + 7.4*c*abs(Ri[i])**0.5)
                fh[i] = 1. - 9.4*Ri[i]/(1. + 5.3*c*abs(Ri[i])**0.5)

        return [fm, fh]




class WageningenU(First_Order_model):
    """
    WageningenU first-order turbulence model (Duynkerke, 1991)
    (derived from "First_Order_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        First_Order_model.__init__(self, dt, nz, Lz, BC, para)


    # (Pseudo) Protected functions:
    def _get_l2(self):
        "Return [lm**2, lh**2]"

        lm = self.BC.k*self.z_secondary[1:-1]
        lm2 = np.multiply(lm,lm)

        return [lm2, lm2]


    def _get_f(self, dUdz2):
        "Return [fm, fh]"

        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)[1:]
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For both stable (Ri>0) and unstable (Ri<0)
        fm = np.zeros_like(Ri)
        fh = np.zeros_like(Ri)
        for i in range(len(Ri)):
            if Ri[i] > 1e-12 :
                if Ri[i] > 1. : # because \psi --> infinity (cut-off)
                    fm[i] = 0.
                    fh[i] = 0.
                else:
                    psi = self.__rootBisect(self.__psiRi_Stable, Ri[i],
                                            0., 1e7)
                    fm[i] = 1./self.__phiStable(psi, 0.8, 5.0)
                    fh[i] = 1./self.__phiStable(psi, 0.8, 7.5)
            elif Ri[i] < -1e-12 :
                if Ri[i] < -1e7 : # As if Ri = -infinity (cut-off)
                    fm[i] = 114.72
                    fh[i] = 11397.
                else:
                    psi = self.__rootBisect(self.__psiRi_UnStable, Ri[i],
                                            -1e7, 0.)
                    fm[i] = 1./self.__phiUnStable(psi, 20., -0.25)
                    fh[i] = 1./self.__phiUnStable(psi, 15., -0.5)
            else:
                fm[i] = 1.
                fh[i] = 1.

        return [np.power(fm, 2), np.multiply(fm, fh) ]


    # Private function
    def __psiRi_Stable(self, psi, Ri):
        return psi*self.__phiStable(psi, 0.8, 7.5) \
                 - Ri*self.__phiStable(psi, 0.8, 5.0)**2


    def __phiStable(self, psi, a, b):
        return 1. + b*psi*(1.+b*psi/a)**(a-1.)


    def __psiRi_UnStable(self, psi, Ri):
        return psi*self.__phiUnStable(psi, 15., -0.5) \
                 - Ri*self.__phiUnStable(psi, 20., -0.25)**2


    def __phiUnStable(self, psi, g, p):
        return (1. - g*psi)**p


    def __rootBisect(self, func, arg, a0, b0, nmax=10000, e=1e-5):
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
