"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
grid.py = staggered grid for turbulence models
 - Turbulence_grid
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np

class Turbulence_grid():
    """
    1D Staggered grid for turbulence models

    Primary grid: u, v, T
    Secondary grid: uw, vw, wT

    Initialization:
    - Number of grid point (nz)
    - Vertical domain (Lz) in m
    - Roughness length (z0) in m (MUST be consistent with SurfaceLayer_BC)
    """
    def __init__(self, nz=64, Lz=400., z0=0.1):
        self._nz = abs(int(nz))
        self._Lz = abs(float(Lz))
        self._z0 = float(z0)

        if self._nz < 3 :
            raise Exception('Turbulence_grid: Too few grid points')

        ### Define
        # Of secondary grid : dz, 2*dz, 3*dz, ... , Lz
        self.z_secondary = np.linspace(0, self._Lz, self._nz+1)[1:]
        self._dz = self.z_secondary[-1] - self.z_secondary[-2] #==z_secondary[1]

        # Of primary grid : z0, 1.5*dz, 2.5*dz, 3.5*dz, ... , Lz-0.5*dz
        self.z_primary = self.z_secondary[:] - 0.5*self._dz
        self.z_primary[0] = self._z0
        self.__dz0_primary = 1.5*self._dz - self._z0

        if self.__dz0_primary <= 0 :
            raise Exception('Turbulence_grid: this grid is not for z0 > 1.5*dz')

        ### Declare
        # (common) Primary variables
        self.u = np.zeros_like(self.z_primary, dtype=float)
        self.v = np.zeros_like(self.z_primary, dtype=float)
        self.T = np.zeros_like(self.z_primary, dtype=float)

        # (common) Secondary variables
        self.uw = np.zeros_like(self.z_secondary, dtype=float)
        self.vw = np.zeros_like(self.z_secondary, dtype=float)
        self.wT = np.zeros_like(self.z_secondary, dtype=float)


    def grad_PtoS(self, X):
        """
        For computing gradient of x = u, v, U, T, ...
        from x on primary grid to dxdz on secondary grid
        (excluding the top BC of the secondary grid)
        """
        return self._grad(X, self.__dz0_primary)


    def grad_StoP(self, X):
        """
        For computing gradient of x = uw, vw, wT, ...
        from x on secondary grid to dxdz on primary grid
        (excluding the bottom BC (surface layer) of the primary grid)
        """
        return self._grad(X, self._dz)


    def onto_P(self, X):
        """
        Interpolate X from secondary grid to primary grid,
        where X = uw, vw, wT, ...
        (excluding the surface z=z0=roughness length)
        """
        return self._interpol(X)


    def onto_S(self, X):
        """
        Interpolate X from primary grid to secondary grid,
        where X = u, v, T, ...
        (excluding the surface layer z=dz and the top BC z=Lz)
        """
        return self._interpol(X[1:])


    # (Pseudo) Protected functions
    def _grad(self, X, dz0):
        "Compute vertical gradient of X"

        dXdz = np.zeros(len(X)-1, dtype=float)

        # Centre finite difference
        dXdz[1:] = (X[2:] - X[1:-1])/self._dz
        dXdz[0] = (X[1]-X[0])/dz0 # May different dz

        return dXdz

    def _interpol(self, X):
        "Simple linear interpolation"
        return 0.5*(X[:-1] + X[1:])



