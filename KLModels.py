"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
KLModels.py = some implementations of k-l turbulence model
- MSC
- KNMI_RACMO
- NASA
- YorkU1
- YorkU2
- LouvainUL
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np

from abstractModels import Setup, KL_model
from BC import Boundary_Cuxart2006 as defaultBC


class MSC(KL_model):
    """
    MSC k-l turbulence model (Belair et al., 1999)
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)

        self.__lm = np.minimum(self.BC.k*self.z_secondary[:-1], 200.)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        return [0.516, 0.516/0.85, 1.0, 0.14] # ceps in stable (Mailhot, 1982)


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        # Ri <-- from Reference paper
        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        dUdz2 = np.power(dudz, 2) + np.power(dvdz, 2)
        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)
        # Ri_f
        S = np.multiply(self.uw[:-1], dudz) + np.multiply(self.vw[:-1], dvdz)
        S = np.maximum(S, 1e-12)
        Rif = np.divide(self.para.g*self.wT[:-1]/self.para.T_ref, S)

        ### For both stable (Ri>0) and unstable (Ri<0)
        fm = np.zeros_like(Ri)
        # Stable
        mask = Ri > 0.
        fm[mask] = 1./(1. + 12.*Ri[mask])
        # Unstable
        mask = np.logical_not(mask)
        fm[mask] = np.power(1. - 40.*Ri[mask], 1./6.)

        feps = np.ones_like(Ri)
        mask = Rif < 0.4
        feps[mask] = np.divide(1.-Rif[mask], 1.-2.*Rif[mask])
        feps[np.logical_not(mask)] = 3.

        return [self.__lm, self.__lm, np.multiply(self.__lm, feps), fm, fm]




class KNMI_RACMO(KL_model):
    """
    KNMI-RACMO k-l turbulence model (Lenderink and Holtslay, 2004)
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)

        self.__lm = 1./(1./self.BC.k/self.z_secondary[:-1] + 1./150.)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        return [0.4868546887, 0.6985192675, 2., 0.071111111111/0.4868546887]


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        dUdz2 = np.power(dudz, 2) + np.power(dvdz, 2)
        dUdz2 = np.maximum(dUdz2, 1e-12)
        dTdz = self.grad_PtoS(self.T)
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        # For stable (Ri>0) ONLY
        if np.any(Ri < 0) :
            print('This model is not for unstable cases yet.')
        temp = np.sqrt(1. + 5.*Ri)
        fm = 1./(1. + 10.*np.divide(Ri, temp))
        fh = 1./(1. + 15.*np.multiply(Ri, temp))

        return [self.__lm, self.__lm, self.__lm, fm, fh]




class NASA(KL_model):
    """
    NASA k-l turbulence model (see Moeng, 1984)
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        return [0.1, 0.1, 2., 1.]


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        # For both stable (Ri>0) and unstable (Ri<0) ?
        Stra = self.para.g*self.grad_PtoS(self.T)/self.para.T_ref

        # Stable
        lm = np.ones_like(Stra)
        mask = Stra > 0
        Stra[mask] = np.maximum(Stra[mask], 1e-12)
        lm[mask] = 0.76*np.sqrt(np.divide(self.TKE[mask], Stra[mask]))
        # Unstable
        mask = np.logical_not(mask)
        lm[mask] = 2*self._dz

        lm = np.minimum(lm, 2*self._dz) # 2, because of staggered grid?
        lh = np.multiply(np.minimum(1. + 2.*lm/self._dz, 3.), lm)
        leps = np.divide(lm, 0.19 + 0.51*lm/self._dz)

        return [lm, lh, leps, np.ones_like(lm), np.ones_like(lm)]




class YorkU1(KL_model):
    """
    1st York University k-l turbulence model (Weng and Taylor, 2003)
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        #return [0.55, 0.55/0.85, 1., 1.] # Cuxart et al (2006)
        cm = (self.BC.cn)**-0.5
        return [cm, cm/0.85, 1., cm**3] # Weng and Taylor (2003) and our BC


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        # For both stable (Ri>0) and unstable (Ri<0)
        Stra = self.para.g*self.grad_PtoS(self.T)/self.para.T_ref

        ### Find lnc
        # 1. rl0 = 1./l0
        rl0 = self.para.f/0.0027/np.sqrt(self.para.Ug**2 + self.para.Vg**2)

        # 2. rlz = phiM/k(z+z0)
        # Assume BC.L>0 for phiM
        if self.BC.rL > 0 :
            phiM = 1. + 4.7*self.BC.rL*self.z_secondary[:-1]
        else:
            phiM = np.power(1. - 15*self.BC.rL*self.z_secondary[:-1], 0.25)
        rlz = np.divide(phiM, self.BC.k*(self.z_secondary[:-1] + self.z_primary[0]))

        # 3. lnc =
        lnc = 1./(rlz + rl0)

        ### Find lm
        lm = np.zeros_like(Stra)
        # Stable
        mask = Stra > 0
        Stra[mask] = np.maximum(Stra[mask], 1e-12)
        lm[mask] = np.minimum(lnc[mask],
                      0.36*np.sqrt(np.divide(self.TKE[mask], Stra[mask])))
        # Unstable
        mask = np.logical_not(mask)
        lm[mask] = lnc[mask]

        return [lm, lm, lm, np.ones_like(lm), np.ones_like(lm)]




class YorkU2(KL_model):
    """
    2nd York University k-l turbulence model
    (Weng and Taylor, 2003) + (Delage, 1974)
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        #return [0.55, 0.55/0.85, 1., 1.] # Cuxart et al (2006)
        cm = (self.BC.cn)**-0.5
        return [cm, cm/0.85, 1., cm**3] # Weng and Taylor (2003) and our BC


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        # For both stable (Ri>0) and unstable (Ri<0)
        if np.any(self.grad_PtoS(self.T) < 0) :
            print('This model is not for unstable cases yet.')

        ### Find lnc
        # 1. rl0 = 1./l0
        rl0 = self.para.f/0.0027/np.sqrt(self.para.Ug**2 + self.para.Vg**2)

        # 2. rlz = phiM/k(z+z0)
        # Assume BC.L>0 for phiM
        if self.BC.rL > 0 :
            phiM = 1. + 4.7*self.BC.rL*self.z_secondary[:-1]
        else:
            phiM = np.power(1. - 15*self.BC.rL*self.z_secondary[:-1], 0.25)
        rlz = np.divide(phiM, self.BC.k*(self.z_secondary[:-1] + self.z_primary[0]))

        # Find lm (stable case only)
        lm = 1./(rlz + rl0 + 4.7*self.BC.rL/0.4)
        leps = 1./(rlz + rl0 + 3.7*self.BC.rL/0.4)

        return [lm, lm, leps, np.ones_like(lm), np.ones_like(lm)]




class LouvainUL(KL_model):
    """
    Louvain University k-l turbulence model
    (derived from "KL_model_S" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        KL_model.__init__(self, dt, nz, Lz, BC, para, scheme)


    # (Pseudo) Protected functions:
    def _get_c(self):
        "Return [cm, ch, ce, deps]"
        return [0.5, 1.3*0.5, 1., 0.125] # ce is guessed.


    def _get_l_and_f(self):
        "Return [lm, lh, leps, fm, fh]"

        ### For stable case (Ri>0) ONLY

        # 1. Find lz
        lz = self.BC.k*self.z_secondary[:-1]

        # 2. Find lsurface
        lsur = 0.3*self.BC.u_star/self.para.f

        # 3. Find ls
        Stra = self.para.g*self.grad_PtoS(self.T)/self.para.T_ref
        if np.any(Stra < 0) :
            print('This model is not for unstable cases yet.')

        ls = np.zeros_like(Stra)
        mask = Stra > 0
        ls[mask] = np.sqrt(self.TKE[mask]/Stra[mask])
        ls[np.logical_not(mask)] = float('inf')

        # 4. Find lm, leps
        lm = 1./(1./lz + 15./lsur + 1.5/ls)
        leps = 1./(1./lz + 15./lsur + 3./ls)

        return [lm, lm, leps, np.ones_like(lm), np.ones_like(lm)]

