"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
IC.py = initial conditions
 - Initial = abstract class for initial_XXX
 - Initial_Cux2006 = class used in computing initial conditions
                     as prescribed in Cuxart et al 2006 (except Eps)
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import abc
import numpy as np


class Initial(abc.ABC):
    """
    Abstract Class that defines functions to calculate initial profiles
    (for our runModels function in tools.py)

    Input: L_eps

    Required:
    - u(model)
    - v(model)
    - T(model)
    - TKE(model)
    - Eps(model) (implemented) = TKE^(3/2)/l
        where 1./l = 1./kz + 1./L_eps
    """
    def __init__(self, L_eps=250.):
        self.L_eps = abs(float(L_eps))


    @abc.abstractmethod
    def u(self):
        pass

    @abc.abstractmethod
    def v(self):
        pass

    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def TKE(self):
        pass

    def Eps(self, model):
        "Return Eps = TKE^{1.5}/l "

        l = 1./(1./model.BC.k/model.z_secondary[:-1] + 1./self.L_eps)
        Eps = np.divide(np.power(self.TKE(model), 3/2), l)

        return Eps




class Initial_Cuxart2006(Initial):
    """
    Class/object that computes initial profiles
    as prescribed in Cuxart et al 2006

    Public parameters:
    - T0 in K
    - dTdz in K
    - L_T in m
    - TKE0 in m^2/s^2
    - L_TKE in m
    """
    def __init__(self, T0=265.0, dTdz=0.01, L_T=100.,
                 TKE0=0.4, L_TKE=250., L_eps=250.):
        Initial.__init__(self, L_eps)
        self.T0 = float(T0)
        self.dTdz = float(dTdz)
        self.L_T = float(L_T)
        self.TKE0 = float(TKE0)
        self.L_TKE = float(L_TKE)


    def u(self, model):
        u = model.para.Ug*np.ones_like(model.z_primary)
        u[0] = 0.

        return u


    def v(self, model):
        return 0*np.ones_like(model.z_primary)


    def T(self, model):
        T = np.zeros_like(model.z_primary)
        z_lower = (model.z_primary <= self.L_T)
        z_upper = np.logical_not(z_lower)

        T[z_lower] = self.T0
        T[z_upper]  = self.T0 \
            + self.dTdz*(model.z_primary[z_upper] - self.L_T)

        return T


    def TKE(self, model):
        TKE = np.zeros_like(model.z_secondary)

        z_lower = (model.z_secondary <= self.L_TKE)
        TKE[z_lower] = self.TKE0 \
            *(1. - model.z_secondary[z_lower]/self.L_TKE)**3
        TKE = np.maximum(TKE, 1e-5)

        return TKE[:-1]

