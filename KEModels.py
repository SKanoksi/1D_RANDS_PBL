"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
KEModels.py = some implementations of k-e turbulence model
- LouvainU-Eps
- Wyngaard
- Engineering
- Test_KE
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np

from abstractModels import Setup, KE_model
from BC import Boundary_Cuxart2006 as defaultBC


class LouvainU_Eps(KE_model):
    """
    LouvainU_eps k-e turbulence model (Duynkerke, 1988)
    (derived from "KE_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=0):
        KE_model.__init__(self, dt, nz, Lz, BC, para)
        self.scheme = int(scheme)

    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, ceps]"
        cm = (self.BC.cn)**-2.
        return [cm, cm, 1./1., 1./2.38]


    def _next_Eps(self, TKE, Eps):
        "Return Eps_{n+1} using the eps equation"

        T_eps = self._Eps_Transport()
        S = 1.46*self._ShearP()
        B = 1.46*np.maximum(self._BuoyP(), 0.)
        T_tke = 1.46*np.maximum(self._TKE_Transport(), 0.)
        D = -1.83*Eps

        # Explicit
        if self.scheme <= 0 :

            RHS = S + B + D + T_tke
            RHS = T_eps + np.multiply(RHS, np.divide(Eps, TKE))
            Eps = Eps + self._dt*RHS

        # Semi-implicit: Eps term
        if self.scheme == 1 :

            RHS = S + B + T_tke
            RHS = T_eps + np.multiply(RHS, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(D, TKE)
            Eps = np.divide(RHS,Div)

        # Semi-implicit: BuoyP, TKE_transport and Eps terms
        if self.scheme >= 2 :

            RHS = T_eps + np.multiply(S, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(B + T_tke + D, TKE)
            Eps = np.divide(RHS,Div)

        return Eps


    # Overridden function
    def Quasi_Eps(self):
        "Return Eps according to the quasi-state approximation"

        BuoyP = self.wT[:-1]*self.para.g/self.para.T_ref
        c2 = np.zeros_like(BuoyP)
        for i in range(len(BuoyP)):
            if BuoyP[i] > 0 :
                c2[i] = 1.46
            else:
                c2[i] = 0.

        return super().Quasi_Eps(1.46, c2, 1.83)




class Wyngaard(KE_model):
    """
    Wyngaard k-e turbulence model (Wyngaard, 1975)
    (also Duynkerke, 1988)
    (derived from "KE_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=0):
        KE_model.__init__(self, dt, nz, Lz, BC, para)
        self.scheme = int(scheme)

    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, ceps]"
        cm = (self.BC.cn)**-2.
        return [cm, cm, 1./1., 0.25/self.BC.cn/0.4/0.4]


    def _next_Eps(self, TKE, Eps):
        "Return Eps_{n+1} using the eps equation"

        T = self._Eps_Transport()
        S = 1.75*self._ShearP()
        B1 = 0.5*self._BuoyP()
        B2 = np.divide(np.power(self._BuoyP(), 2), Eps)
        D = -2.*Eps

        # Explicit
        if self.scheme <= 0 :

            RHS = S + B1 + B2 + D
            RHS = T + np.multiply(RHS, np.divide(Eps, TKE))
            Eps = Eps + self._dt*RHS

        # Semi-implicit: Eps term
        if self.scheme == 1 :

            RHS = S + B1 + B2
            RHS = T + np.multiply(RHS, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(D, TKE)
            Eps = np.divide(RHS,Div)

        # Semi-implicit: BuoyP (B1 as B2 = positive) and Eps terms
        if self.scheme >= 2 :

            RHS = T + np.multiply(S + B2, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(B1 + D, TKE)
            Eps = np.divide(RHS,Div)

        return Eps

    # Overridden function
    def Quasi_Eps(self):
        "Return Eps according to the quasi-steady state approximation"

        BuoyP = self.wT[:-1]*self.para.g/self.para.T_ref
        B1 = 0.5
        B2 = np.divide(BuoyP, self.Eps)

        return super().Quasi_Eps(1.75, B1+B2, 2.)




class Engineering(KE_model):
    """
    Standard k-e turbulence model in Engineering (Junfu et al, 2016)

    (derived from "KE_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=0):
        KE_model.__init__(self, dt, nz, Lz, BC, para)
        self.scheme = int(scheme)

    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, ceps]"
        cm = (self.BC.cn)**-2.
        return [cm, cm, 1./1., 1./1.3]


    def _next_Eps(self, TKE, Eps):
        "Return Eps_{n+1} using the eps equation"

        T = self._Eps_Transport()
        S = 1.44*self._ShearP()
        B = 1.44*0.09*self._BuoyP()
        D = -1.92*Eps

        # Explicit
        if self.scheme <= 0 :

            RHS = S + B + D
            RHS = T + np.multiply(RHS, np.divide(Eps, TKE))
            Eps = Eps + self._dt*RHS

        # Semi-implicit: Eps term
        if self.scheme == 1 :

            RHS = S + B
            RHS = T + np.multiply(RHS, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(D, TKE)
            Eps = np.divide(RHS,Div)

        # Semi-implicit: BuoyP and Eps terms
        if self.scheme >= 2 :

            RHS = T + np.multiply(S, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(B + D, TKE)
            Eps = np.divide(RHS,Div)

        return Eps


    # Overridden function
    def Quasi_Eps(self):
        "Return Eps according to the quasi-steady state approximation"
        return super().Quasi_Eps(1.44, 1.44*0.09, 1.92)




class Test_KE(KE_model):
    """
    Our test k-e turbulence model

    (derived from "KE_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=0):
        KE_model.__init__(self, dt, nz, Lz, BC, para)
        self.scheme = int(scheme)

    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, ceps]"

        cm = (self.BC.cn)**-2.
        ceps = 0.5

        return [cm, cm, 1., ceps]


    def _next_Eps(self, TKE, Eps):
        "Return Eps_{n+1} using the eps equation"

        c_1 = 1.4  # From homogeneous shear flow ### Checked
        c_2 = 0.2
        c_3 = 1.3
        c_4 = 1.8  # From decay of turbulence ### Checked

        T_eps = self._Eps_Transport()
        S = c_1*self._ShearP()
        B = np.multiply(c_2, self._BuoyP())
        T_tke = c_3*self._TKE_Transport()
        D = -c_4*Eps

        # Explicit
        if self.scheme <= 0 :

            RHS = S + B + T_tke + D
            RHS = T_eps + np.multiply(RHS, np.divide(Eps, TKE))
            Eps = Eps + self._dt*RHS

        # Semi-implicit: Eps term
        if self.scheme == 1 :

            RHS = S + B + T_tke
            RHS = T_eps + np.multiply(RHS, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(D, TKE)
            Eps = np.divide(RHS,Div)

        # Semi-implicit: BuoyP, TKE_transport and Eps terms
        if self.scheme >= 2 :

            RHS = T_eps + np.multiply(S, np.divide(Eps, TKE))
            RHS = Eps + self._dt*RHS
            Div = 1. - self._dt*np.divide(B + T_tke + D, TKE)
            Eps = np.divide(RHS,Div)


        return Eps


    # Overridden function
    def Quasi_Eps(self):
        "Return Eps according to the quasi-steady state approximation"

        c_1 = 1.4
        c_2 = 0.2
        c_4 = 1.8 # From decay of turbulence ### Checked

        return super().Quasi_Eps(c_1, c_2, c_4)
