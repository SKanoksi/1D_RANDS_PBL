"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
abstractModels.py = Abstract base classes (and auxiliary classes)
 - Setup = contain common parameters
 - Turbulence_model = Base class for the following abstract models
     + First_Order_model
     + KL_model (S)
     + KE_model (S)
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import abc
import numpy as np

from BC import Boundary_Cuxart2006 as defaultBC
from grid import Turbulence_grid

class Setup():
    """
    Class that contains parameters for setting up our turbulence models

    Public parameters:
    - Constant geostrophic wind ([Ug,Vg]) in m/s
    - Coriolis parameter (f) in /s
    - Reference potential temperature (T_ref) in K
    - Reference density (rho_ref) in kg/m^3
    - Gravitational constant (g) in m/s^2
    """
    def __init__(self, Ug=8., Vg=0., f=1.39e-4,
                 T_ref=263.5, rho_ref=1.3223, g=9.81):
        self.Ug = float(Ug)
        self.Vg = float(Vg)
        self.f  = float(f)
        self.T_ref = float(T_ref)
        self.rho_ref = abs(float(rho_ref))
        self.g = abs(float(g))




class Turbulence_model(abc.ABC, Turbulence_grid):
    """
    Abstract Class that defines main functions
    with a default simple FT method

    Initialization:
    - Timestep size (dt) in s
    - Number of grid point (nz), including at z=z0
    - Vertical domain (Lz) in m
    - Boundary conditions (BC) = class that contains required functions
    - para = the Setup class containing parameters
    Public functions:
    - run() = forward one time step
    - time() = return simulation time in hour
    """
    def __init__(self, dt=10., nz=65, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        Turbulence_grid.__init__(self, nz, Lz, BC.z0)
        self._dt = abs(float(dt))
        self.BC = BC
        self.para = para

        self._it = 0


    def run(self):
        "Forward one iteration using FT (Turbulence_model)"

        # Update stresses at n
        self._update_stresses()

        # Apply BCs at n
        self.BC.update_surface_uvT(self)
        self.BC.update_surface_stresses(self)
        self.BC.update_top_stresses(self)

        # Forward u,v,T in time from n to n+1
        self._update_uvT()
        self._it += 1


    def time(self):
        "Current (total) simulation time in hour"
        return self._it*self._dt/3600


    # (Pseudo) Protected functions:
    @abc.abstractmethod
    def _eddyViscosity(self):
        pass


    def _update_uvT(self):
        "Forward FD in time u,v,T"

        temp = self.u[1:] + self._dt*self.__sourceU()
        self.v[1:] = self.v[1:] + self._dt*self.__sourceV()
        self.T[1:] = self.T[1:] + self._dt*self.__sourceT()
        self.u[1:] = temp.copy()
        # (No u,v directly in __sourceT)


    def _update_stresses(self):
        "Update uw,vw,wT"

        Km, Kh = self._eddyViscosity()
        self.uw[1:-1] = -np.multiply(Km, self.grad_PtoS(self.u)[1:] )
        self.vw[1:-1] = -np.multiply(Km, self.grad_PtoS(self.v)[1:] )
        self.wT[1:-1] = -np.multiply(Kh, self.grad_PtoS(self.T)[1:] )


    # Private functions:
    def __sourceU(self):
        "RHS of the equation for u"
        return self.para.f*(self.v[1:] - self.para.Vg) \
                - self.grad_StoP(self.uw)


    def __sourceV(self):
        "RHS of the equation for v"
        return -self.para.f*(self.u[1:] - self.para.Ug) \
                - self.grad_StoP(self.vw)


    def __sourceT(self):
        "RHS of the equation for T"
        return -self.grad_StoP(self.wT)




class First_Order_model(Turbulence_model):
    """
    Abstract Class for first-order models
    (derived from "Turbulence_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        Turbulence_model.__init__(self, dt, nz, Lz, BC, para)

        # Get the mixing lengthes (squared)
        [self.__lm2, self.__lh2] = self._get_l2()


    # (Pseudo) Protected functions
    def _eddyViscosity(self):
        "Return Km, Kh (first-order model)"

        # Compute |d\vec{U}dz|
        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        dUdz = ( np.power(dudz,2) + np.power(dvdz,2) )[1:]
        # Subclass calculates stability functions
        [fm, fh] = self._get_f(dUdz)
        dUdz = np.sqrt(dUdz)

        # Estimating eddy viscosity
        Km = np.multiply(self.__lm2, np.multiply(dUdz, fm))
        Kh = np.multiply(self.__lh2, np.multiply(dUdz, fh))

        return Km, Kh


    @abc.abstractmethod
    def _get_l2(self):
        pass


    @abc.abstractmethod
    def _get_f(self):
        pass




class KL_model(Turbulence_model):
    """
    Abstract Class for k-l model
    where TKE is on secondary grid
    (derived from "Turbulence_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup(), scheme=2):
        Turbulence_model.__init__(self, dt, nz, Lz, BC, para)
        self.Scheme = int(scheme)

        ### wE on primary grid
        self.wE = np.zeros(len(self.z_primary)-1, dtype=float)
        # (wE = excluding bottom BC)

        ### Km, leps, TKE on secondary grid
        n = len(self.z_secondary) - 1
        self._Km = np.zeros(n, dtype=float)
        self.leps = np.zeros(n, dtype=float)
        self.TKE = np.zeros(n, dtype=float)
        # (Km, leps, TKE = excluding top BC)

        # Get the parameters
        [self.__cm, self.__ch, self.__ce, self._deps] = self._get_c()


    # Overridden function
    def run(self):
        "Forward one iteration (KL_model (S))"

        # Update stresses and BCs at n
        self.BC.update_surface_uvT(self)
        self.BC.update_surface_stresses(self)
        self._update_stresses() # Need TKE[1:] at n
        self.__update_wE() # Need Km, full TKE at n
        self.BC.update_top_stresses(self) # Need stresses at n (no-flux BC)

        # Forward u,v,T, TKE in time from n to n+1
        self.__update_TKE() # Need wE at n
        self._update_uvT() # Need T at n
        self._it += 1

        # Finish with
        # at n+1 : u,v,T, TKE[1:], it
        # at n : uw, vw, wT, wE, TKE[0], Km, Kh


    # (Pseudo) Protected functions
    def _eddyViscosity(self):
        "Return Km, Kh (KL model (S))"

        # Subclass calculates length scales and stability functions
        [lm, lh, self.leps, fm, fh] = self._get_l_and_f()

        # Estimating eddy viscosity
        self._Km = self.__cm*np.multiply(np.sqrt(self.TKE),
                                   np.multiply(lm, fm))
        Kh = self.__ch*np.multiply(np.sqrt(self.TKE),
                                   np.multiply(lh, fh))

        return self._Km[1:], Kh[1:]


    @abc.abstractmethod
    def _get_c(self):
        pass


    @abc.abstractmethod
    def _get_l_and_f(self):
        pass


    # Private functions
    def __update_wE(self):
        "Update wE for TKE equation"
        self.wE[:-1] = -self.__ce*np.multiply(
                            self.onto_P(self._Km),
                            self.grad_StoP(self.TKE))


    def __update_TKE(self):
        "Update the TKE equation"

        # Three terms on RHS
        ShearP = -np.multiply(self.uw[1:-1], self.grad_PtoS(self.u)[1:]) \
                 -np.multiply(self.vw[1:-1], self.grad_PtoS(self.v)[1:])
        BuoyP = self.wT[1:-1]*self.para.g/self.para.T_ref
        Transport = -self._grad(self.wE, self._dz) # == FD, equally spacing dz

        # Time-stepping
        ### 0. Explicit scheme: BruteForce 0<TKE
        if self.Scheme <= 0 :

            # Eps = d*(e_{n}^1.5)/leps
            Eps = self._deps*np.divide(np.power(self.TKE[1:], 1.5),
                                         self.leps[1:])
            # Update
            self.TKE[1:] += self._dt*(ShearP + Transport - Eps)
            # (BruteForce!!) BuoyP must not make TKE negative
            self.TKE[1:] = np.maximum(self.TKE[1:] + self._dt*BuoyP, 0.)

        ### 1.Semi-implicit scheme 1: BruteForce 0<TKE
        if self.Scheme == 1 :

            # Eps = -d*(e_{n+1}^1 * e_{n}^0.5)/leps
            Div = 1. + self._dt*self._deps \
                        *np.divide(np.sqrt(self.TKE[1:]), self.leps[1:])
            # Update
            self.TKE[1:] = np.divide(self.TKE[1:]
                            + self._dt*(ShearP + Transport), Div)
            # (BruteForce!!) BuoyP must not make TKE negative
            self.TKE[1:] = np.maximum(self.TKE[1:]
                            + np.divide(self._dt*BuoyP, Div), 0.)

        ### 2. Semi-implicit scheme 2: Implicit BuoyP and Eps
        if self.Scheme == 2 :

            # BuoyP = BuoyP*(e_{n+1}/e_{n})
            # Eps = -d*(e_{n+1}^1 * e_{n}^0.5)/leps
            Div = 1. - self._dt*( \
                        np.divide(BuoyP, self.TKE[1:])
                       - self._deps*np.divide(np.sqrt(self.TKE[1:]),
                                               self.leps[1:]))
            # Update
            self.TKE[1:] = np.divide(self.TKE[1:]
                            + self._dt*(ShearP + Transport), Div)
            # This scheme can be unstable if Div = effectively 0

        ### 3. Semi-implicit scheme 3: Implicit BuoyP and Eps
        if self.Scheme >= 3 :

            # BuoyP = BuoyP*(e_{n+1}^0.5/e_{n}^0.5)
            # Eps = -d*(e_{n}^1 * e_{n+1}^0.5)/leps
            Div = np.sqrt(self.TKE[1:]) - self._dt*( \
                        np.divide(BuoyP, np.sqrt(self.TKE[1:]))
                       - self._deps*np.divide(self.TKE[1:], self.leps[1:]))

            # Update
            self.TKE[1:] = np.divide(self.TKE[1:]
                            + self._dt*(ShearP + Transport), Div)
            self.TKE[1:] = np.power(self.TKE[1:], 2)
            # This scheme can be unstable due to sqrt(TKE) --> effectively 0


        # Check (Negative TKE give K=nan)
        if np.any(self.TKE < 0) or np.any(np.logical_not(np.isfinite(self.TKE))) :
            print('Some invalid TKE at iteration', self._it, ', time', self.time())
            # --- Add below to examine the cause ---


    def Quasi_TKE(self):
        "Return TKE according to the quasi-steady state approximation"

        """
        # Won't work due to small uw --> wT/uw  = large error
        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        div = np.multiply(self.uw[:-1], dudz) + np.multiply(self.vw[:-1], dvdz)
        div = np.maximum(div, 1e-12)
        Rif = np.divide(self.para.g*self.wT[:-1]/self.para.T_ref, div)
        """

        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        dUdz2 = np.maximum(np.power(dudz,2) + np.power(dvdz,2), 1e-16)
        dTdz = self.grad_PtoS(self.T)
        Ri = np.divide(self.para.g*dTdz/self.para.T_ref, dUdz2)

        [lm, lh, leps, fm, fh] = self._get_l_and_f()
        lm = np.multiply(lm, fm)
        lh = np.multiply(lh, fh)
        leps = leps/self._deps

        Pr = self.__cm*lm/self.__ch/lh
        Pr = np.maximum(Pr, 1e-16)

        Quasi_TKE = np.multiply(lm, leps)*self.__cm
        Quasi_TKE = np.multiply(Quasi_TKE, np.power(dudz,2) + np.power(dvdz,2))
        Quasi_TKE = np.multiply(Quasi_TKE, 1.-np.divide(Ri, Pr))

        # Only positive TKE
        Quasi_TKE = np.maximum(Quasi_TKE, 0.)

        # At boundary
        Quasi_TKE[0] = self.BC._surfaceTKE(self.z_secondary[0])

        return Quasi_TKE




class KE_model(Turbulence_model):
    """
    Abstract Class for k-e model
    where TKE and eps are on "secondary" grid
    (derived from "Turbulence_model" class)
    """
    def __init__(self, dt=10., nz=64, Lz=400.,
                 BC=defaultBC(), para=Setup()):
        Turbulence_model.__init__(self, dt, nz, Lz, BC, para)

        ### wE, wEps on primary grid
        n = len(self.z_primary) - 1
        self.wE = np.zeros(n, dtype=float)
        self.wEps = np.zeros(n, dtype=float)
        # (wE, wEps = excluding bottom BC)

        ### Km, TKE, Eps on secondary grid
        n = len(self.z_secondary) - 1
        self._Km = np.zeros(n, dtype=float)
        self.TKE = np.zeros(n, dtype=float)
        self.Eps = np.zeros(n, dtype=float)
        # (Km, TKE, Eps = excluding top BC)

        # Get the parameters
        [self.__cm, self.__ch, self.__ce, self.__ceps] = self._get_c()
        # (ceps in k-e model != ceps in k-l model)


    # Overridden function
    def run(self):
        "Forward one iteration (KE_model)"

        # Update stresses and BCs at n
        self.BC.update_surface_uvT(self)
        self.BC.update_surface_stresses(self)
        self._update_stresses() # Need TKE[1:], Eps[1:] at n
        self.__update_wE_and_wEps() # Need Km, full TKE, full Eps at n
        self.BC.update_top_stresses(self) # Need stresses at n (no-flux BC)

        # Forward u,v,T, TKE, Eps in time from n to n+1
        self.__update_TKE_and_Eps() # Need wE, wEps at n
        self._update_uvT() # Need T at n
        self._it += 1

        # Finish with
        # at n+1 : u,v,T, TKE[1:], it
        # at n : uw, vw, wT, wE, TKE[0], Km, Kh


    # (Pseudo) Protected functions
    def _eddyViscosity(self):
        "Return Km, Kh (KE model)"

        # Estimating eddy viscosity
        self._Km = self.__cm*np.divide(np.power(self.TKE, 2), self.Eps)
        Kh = self.__ch*np.divide(np.power(self.TKE, 2), self.Eps)

        return self._Km[1:], Kh[1:]


    @abc.abstractmethod
    def _get_c(self):
        pass


    @abc.abstractmethod
    def _next_Eps(self, TKE, eps):
        pass


    # Functions for sub-class
    def _ShearP(self):
        "Return shear production term in TKE equation"
        return -np.multiply(self.uw[1:-1], self.grad_PtoS(self.u)[1:]) \
               -np.multiply(self.vw[1:-1], self.grad_PtoS(self.v)[1:])


    def _BuoyP(self):
        "Return Buoyant production term in TKE equation"
        return self.wT[1:-1]*self.para.g/self.para.T_ref


    def _TKE_Transport(self):
        "Return TKE transport term in TKE equation"
        return -self._grad(self.wE, self._dz)


    def _Eps_Transport(self):
        "Return Eps transport term in Eps equation"
        return -self._grad(self.wEps, self._dz)


    # Private functions
    def __update_wE_and_wEps(self):
        "Update wE for TKE equation and wEps for Eps equation"
        self.wE[:-1] = -self.__ce*np.multiply(
                            self.onto_P(self._Km),
                            self.grad_StoP(self.TKE))
        self.wEps[:-1] = -self.__ceps*np.multiply(
                            self.onto_P(self._Km),
                            self.grad_StoP(self.Eps))


    def __update_TKE_and_Eps(self):
        "Update the TKE and eps equations (in Normal form)"

        TKE = self.TKE.copy()[1:]
        Eps = self.Eps.copy()[1:]

        # RHS
        RHS_TKE = self._ShearP() + self._BuoyP() \
                    + self._TKE_Transport() \
                    - Eps

        # Update
        self.TKE[1:] += self._dt*RHS_TKE # Simple FD
        self.Eps[1:] = self._next_Eps(TKE, Eps)


    def Quasi_Eps(self, c1, c2, c4):
        "Return Eps according to the quasi-steady state approximation"

        dudz = self.grad_PtoS(self.u)
        dvdz = self.grad_PtoS(self.v)
        Shear = (np.power(dudz, 2) + np.power(dvdz, 2))
        Stra = self.para.g*self.grad_PtoS(self.T)/self.para.T_ref

        div = c1*Shear - np.multiply(c2, Stra)
        div = np.maximum(div, 1e-12)
        leps = np.zeros_like(div)
        # Ri < Ric (where Ric = c1/c2)
        mask = div > 0
        leps[mask] = np.sqrt(c4*self.TKE[mask]/self.__cm/div[mask])
        # Ri > Ric
        leps[np.logical_not(mask)] = float('inf')

        # At boundary
        TKE = self.BC._surfaceTKE(self.z_secondary[0])
        eps = self.BC._surfaceEps(self.z_secondary[0])
        leps[0] = TKE**1.5/eps

        quasi_eps = np.divide(np.power(self.TKE, 3/2), leps)

        return quasi_eps, leps
