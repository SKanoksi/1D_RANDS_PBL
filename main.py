"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
Main.py = several main functions
- mainStable()
- mainBC()
- mainVaryKE()
- mainQuasi()
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

from FirstOrderModels import ECMWF, MeteoFrance, NOAA_NCEP, JMA, \
                             MetOffice, WageningenU
from KLModels import MSC, KNMI_RACMO, NASA, YorkU1, YorkU2, LouvainUL
from KEModels import LouvainU_Eps, Wyngaard, Engineering, Test_KE
from abstractModels import Setup
from IC import Initial_Cuxart2006
from BC import Boundary_Cuxart2006
from tools import runModels, runQuasi


def mainStable(run_firstorder=True, run_kl=True, run_ke=True):

    nt = 9*60*60
    dt = 0.8
    nz = 64
    Lz = 400.
    defaultBC = lambda logForm=True : Boundary_Cuxart2006(logForm=logForm)
    defaultIC = lambda : Initial_Cuxart2006()
    para = Setup()

    # ***********************
    #  First-order models
    # ***********************
    if run_firstorder :
        name_list = ['ECMWF-MO','MeteoFrance','MetOffice']
        model_list = name_list.copy()
        model_list[0] = ECMWF(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
        model_list[1] = MeteoFrance(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
        model_list[2] = MetOffice(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
        runModels(nt, model_list, name_list, init=defaultIC(), style='-')

        name_list = ['NOAA-NCEP','JMA']
        model_list = name_list.copy()
        model_list[0] = NOAA_NCEP(dt=dt, nz=nz, Lz=Lz,
                                BC=defaultBC(logForm=False), para=para)
        model_list[1] = JMA(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
        runModels(nt, model_list, name_list, init=defaultIC(), style='-')

        name_list = ['WageningenU']
        model_list = name_list.copy()
        model_list[0] = WageningenU(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
        runModels(nt, model_list, name_list, init=defaultIC(), style='-')

    # ****************************************
    #  k-L models that TKE on secondary grid
    # ****************************************

    if run_kl :
        name_list = ['MSC','KNMI_RACMO', 'NASA','YorkU-1','YorkU-2','LouvainU-L']
        model_list = name_list.copy()
        model_list[0] = MSC(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=2)
        model_list[1] = KNMI_RACMO(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=2)
        model_list[2] = NASA(dt=dt, nz=nz, Lz=Lz,
                            BC=defaultBC(logForm=False), para=para,
                            scheme=2)
        model_list[3] = YorkU1(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=2)
        model_list[4] = YorkU2(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=0)
        model_list[5] = LouvainUL(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=2)
        runModels(nt, model_list, name_list, init=defaultIC(), style='-')

    # ******************************************************
    #  k-e models that TKE and eps are on secondary grid
    # ******************************************************

    if run_ke :
        name_list = ['LouvainU-Eps','Wyngaard','Engineering']
        model_list = name_list.copy()
        model_list[0] = LouvainU_Eps(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=0)
        model_list[1] = Wyngaard(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=0)
        model_list[2] = Engineering(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                            scheme=0)
        runModels(nt, model_list, name_list, init=defaultIC(), style='--')

    return


def mainBC():

    nt = 9*60*60
    dt = 0.8
    nz = 32
    Lz = 400
    defaultBC = lambda : Boundary_Cuxart2006(logForm=False)
    defaultIC = lambda : Initial_Cuxart2006()
    para = Setup()

    # ******************************************************
    #  Comparing surface layer schemes
    # ******************************************************

    name_list = ['MetOffice: dz = 12.5']
    model_list = name_list.copy()
    model_list[0] = MetOffice(dt=dt, nz=nz, Lz=Lz,
                           BC=defaultBC(), para=para)
    runModels(nt, model_list, name_list, init=defaultIC(), style='r--')

    name_list = ['YorkU1: dz = 12.5']
    model_list = name_list.copy()
    model_list[0] = YorkU1(dt=dt, nz=nz, Lz=Lz,
                           BC=defaultBC(), para=para,
                           scheme=2)
    runModels(nt, model_list, name_list, init=defaultIC(), style='g--')

    name_list = ['Wyngaard: dz = 12.5']
    model_list = name_list.copy()
    model_list[0] = Wyngaard(dt=dt, nz=nz, Lz=Lz,
                           BC=defaultBC(), para=para,
                           scheme=0)
    runModels(nt, model_list, name_list, init=defaultIC(), style='b--')

    return


def mainVaryKE():

    nt = 9*60*60
    dt = 0.8
    nz = 64
    Lz = 400.
    defaultBC = lambda logForm=True : Boundary_Cuxart2006(logForm=logForm)
    defaultIC = lambda : Initial_Cuxart2006()
    para = Setup()

    # ******************************************************
    #  Varing constants in the K-E model
    # --> Manually vary constants of Test_KE(KE_model) class
    #     in KEModels.py !!!
    # ******************************************************

    style = 'b-'
    name_list = ['C4 = 1.8']
    model_list = name_list.copy()
    model_list[0] = Test_KE(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para)
    runModels(nt, model_list, name_list, init=defaultIC(), style=style)

    return


def mainQuasi():

    nt = 9*60*60
    dt = 0.8
    nz = 64
    Lz = 400.
    defaultBC = lambda logForm=True : Boundary_Cuxart2006(logForm=logForm)
    defaultIC = lambda : Initial_Cuxart2006()
    para = Setup()

    # ****************************************
    #  Quasi TKE of k-L models
    # ****************************************

    name_list = ['MSC','KNMI_RACMO', 'NASA','YorkU-1','YorkU-2','LouvainU-L']
    model_list = name_list.copy()
    model_list[0] = MSC(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                           scheme=2)
    model_list[1] = KNMI_RACMO(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                           scheme=2)
    model_list[2] = NASA(dt=dt, nz=nz, Lz=Lz,
                           BC=defaultBC(logForm=False), para=para,
                           scheme=2)
    model_list[3] = YorkU1(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                           scheme=2)
    model_list[4] = YorkU2(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                           scheme=0)
    model_list[5] = LouvainUL(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                           scheme=2)
    runQuasi('TKE', nt, model_list, name_list, init=defaultIC(), style='-')

    # ******************************************************
    #  Quasi Eps of k-e models
    # ******************************************************

    name_list = ['LouvainU-Eps','Wyngaard','Engineering']
    model_list = name_list.copy()
    model_list[0] = LouvainU_Eps(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                          scheme=0)
    model_list[1] = Wyngaard(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                          scheme=0)
    model_list[2] = Engineering(dt=dt, nz=nz, Lz=Lz, BC=defaultBC(), para=para,
                          scheme=0)
    runQuasi('Eps', nt, model_list, name_list, init=defaultIC(), style='-')

    return




### Main experiments
# Warning!! Every run will plot/overlay on the same figures.

#mainStable(run_firstorder=True, run_kl=False, run_ke=False)
mainStable(run_firstorder=False, run_kl=True, run_ke=True)
#mainBC()
#mainVaryKE()
#mainQuasi()

