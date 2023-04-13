"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
BC.py = boundary conditions
 - Boundary = abstract class for Boundary_XXX
 - Boundary_Cux2006 = class used in computing boundary conditions
                      as prescribed in Cuxart et al 2006
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import abc
import numpy as np

class Boundary(abc.ABC):
    """
    Abstract Class that defines functions for updating BC
    (for BC in our turbulence models)

    Required:
    - update_surface_uvT(model) = no-slip condition
    - update_top_stresses(model) = no-flux condition
    - update_surface_stresses(model) = (Integrated) similarity theory
    - _surfaceTKE() = (Andre et al, 1978)
    - _surfaceEps() = (Eps = S + B)
    Need to be overridden:
    - surfaceT(time) = prescribed surface (potential) temperature
    - phi_m(zeta)
    - phi_h(zeta)
    - psi_m(zeta)
    - psi_h(zeta)
    """
    def __init__(self, z0=0.1, cn=3.75, k=0.4):
        self.z0 = abs(float(z0))
        self.k  = float(k)
        self.cn = float(cn)

        self.rL = 0. # = 1/L and assume initially neutral condition at t=0
        self.u_star = 1e12


    def update_surface_uvT(self, model):
        "Update no-slip BC and Tsurface(time) of input model"

        model.u[0] = 0. # Actually no need as untouched
        model.v[0] = 0. # Actually no need as untouched
        model.T[0] = self.surfaceT(model.time())


    def update_top_stresses(self, model):
        "Specify no-flux BC at the top"

        # No-flux == No-stress == at the top on primary grid
        model.uw[-1] = 0.  # Actually no need as untouched
        model.vw[-1] = 0.  # Actually no need as untouched
        model.wT[-1] = 0.  # Actually no need as untouched

        # No-flux boundary condition at the top (First-order BC)
        if hasattr(model, 'wE') :
            model.wE[-1] = 0.   # Actually no need as untouched
        if hasattr(model, 'wEps') :
            model.wEps[-1] = 0. # Actually no need as untouched


    def update_surface_stresses(self, model):
        "Update surface layer stresses of input model"

        stress = self._SurfaceStress(model)
        model.uw[0] = stress[0]
        model.vw[0] = stress[1]
        model.wT[0] = stress[2]
        if hasattr(model, 'TKE') :
            model.TKE[0] = self._surfaceTKE(model.z_secondary[0])
        if hasattr(model, 'Eps') :
            model.Eps[0] = self._surfaceEps(model.z_secondary[0])


    # (Pseudo) Private functions
    def _SurfaceStress(self, model):
        "Return [uw0, vw0, wT0, 1/L, u_star, T_star]"

        ### Log form, where L at nt-1 --> u_star, T_star
        # Log form = better as wind/temp change swiftly near surface (log func)
        # -> Therefore, log/integral form can cope with them than grad/diff form
        # -> Scale independent
        # -> Weakness = L estimated from previous step

        # 1. Find u_star, Theta_star
        z0 = model.z_primary[0] # Bottom of the surface layer
        z  = model.z_primary[1] # Top of the surface layer
        # where model.z_secondary[0] = in the surface layer
        U  = (model.u[0:2]**2 + model.v[0:2]**2)**0.5
        # u*, T*
        self.u_star = (U[1] - U[0])*self.k \
            /(np.log(z/z0) - self.psi_m(z*self.rL) + self.psi_m(z0*self.rL))
        T_star = (model.T[1] - model.T[0])*self.k \
            /(np.log(z/z0) - self.psi_h(z*self.rL) + self.psi_h(z0*self.rL))
        # Use log(x) where x is dimensionless -> Otherwise = blow up easily


        # 2. Convert to turbulent stresses
        dudz = model.grad_PtoS(model.u[0:2])
        dvdz = model.grad_PtoS(model.v[0:2])
        div = max(model.grad_PtoS(U), 1e-12)
        # uw, vw, wT
        uw = -self.u_star**2*abs(dudz/div)
        vw = -self.u_star**2*abs(dvdz/div)
        wT = -self.u_star*T_star

        # 3. 1/L and Save as required by surface TKE, Eps and some models
        self.rL = self.k*model.para.g*T_star \
                    /max(model.para.T_ref*self.u_star**2, 1e-12)

        return [uw, vw, wT, self.rL, self.u_star, T_star]


    def _surfaceTKE(self, z):
        "Surface TKE prescribed in Andre et al (1978)"

        # If Unstable, else Stable
        if self.rL < 0 :
            w_star = self.u_star*(-z*self.rL/self.k)**(1/3)
            return (self.cn + (-z*self.rL)**(-2/3))*self.u_star**2 \
                    + 0.3*w_star**2
        else:
            return self.cn*self.u_star**2


    def _surfaceEps(self, z):
        "Return Surface Eps at z (Assuming Eps = ShearP + BouyP)"

        # = (Shear + Buoy)/(u_star**3)
        coeff = self.phi_m(z*self.rL)/self.k/z - self.rL/self.k

        return coeff*self.u_star**3


    @abc.abstractmethod
    def surfaceT(self, time):
        pass

    @abc.abstractmethod
    def phi_m(self, zeta):
        pass

    @abc.abstractmethod
    def phi_h(self, zeta):
        pass

    @abc.abstractmethod
    def psi_m(self, zeta):
        pass

    @abc.abstractmethod
    def psi_h(self, zeta):
        pass




class Boundary_Cuxart2006(Boundary):
    """
    Class/object that computes boundary conditions
    as prescribed in Cuxart et al 2006
    (an input to our turbulence model)

    Public parameters:
    - T0 = initial surface potential temperature in K
    - dTdt = rate in K/hour
    - Roughness length (z0) in metre
    - logForm = surface similarity thoory in logarithm form or differntial form
    """
    def __init__(self, T0_init=265., dTdt=-0.25, logForm=True,
                 z0=0.1, cn=5.5, k=0.4):
        Boundary.__init__(self, z0=z0, cn=cn, k=k)
        self.logForm = logForm

        self.T0_init = float(T0_init)
        self.dTdt = float(dTdt)


    def surfaceT(self, time):
        return self.T0_init + self.dTdt*time


    def phi_m(self, zeta):
        if zeta < 0 :
            print('This BC class is for stable case only.')

        return 1. + 4.8*zeta


    def phi_h(self, zeta):
        if zeta < 0 :
            print('This BC class is for stable case only.')

        return 1. + 7.8*zeta


    def psi_m(self, zeta):
        if zeta < 0 :
            print('This BC class is for stable case only.')

        return - 4.8*zeta


    def psi_h(self, zeta):
        if zeta < 0 :
            print('This BC class is for stable case only.')

        return - 7.8*zeta


    # Overridden function
    def _SurfaceStress(self, model):
        "Return [uw0, vw0, wT0, 1/L, u_star, T_star]"

        if self.logForm :
            return super()._SurfaceStress(model)
        else:
            return self.__SurfaceStress_Diff(model)


    # (Unconventional version)
    def __SurfaceStress_Diff(self, model):
        """
        Compute surface stresses in Stable condition (Only)
        """

        ### Differential form and L at nt --> Solve analytically
        # Weakness -> Scale dependent due to FD
        # Strength -> L at the present timestep
        beta_m = 4.8
        beta_h = 7.8

        # 0. Compute gradients
        z = (model.z_primary[0] + model.z_primary[1])/2
        U  = (model.u[0:2]**2 + model.v[0:2]**2)**0.5
        dUdz = max(model.grad_PtoS(U), 1e-12)
        dTdz = model.grad_PtoS(model.T[0:2])

        # 1. Compute R=T_star/u_star, u_star and T_star analytically
        para = model.para
        const = (beta_h - beta_m)*para.g/para.T_ref
        R = (np.sqrt(dUdz**2 + 4*dTdz*const) - dUdz)/(2*const)
        self.u_star = (dUdz - R*beta_m*para.g/para.T_ref)*self.k*z
        T_star = R*self.u_star

        # 2. Convert to turbulent stresses
        # uw, vw
        dudz = model.grad_PtoS(model.u[0:2])
        dvdz = model.grad_PtoS(model.v[0:2])
        uw = -self.u_star**2*abs(dudz/dUdz)
        vw = -self.u_star**2*abs(dvdz/dUdz)
        # wT
        wT = -self.u_star*T_star

        # 3. 1/L and Save as required by surface TKE, Eps and some models
        self.rL = self.k*model.para.g*T_star \
                    /max(model.para.T_ref*self.u_star**2, 1e-12)

        return [uw, vw, wT, self.rL, self.u_star, T_star]
