"""
----------------------------------
Turbulence models
MSc Project @ Reading:
    Somrath Kanoksirirath
    StudentID 26835996
----------------------------------
tools.py = various tools
- runModels
----------------------------------
 Copyright (c) 2019, Somrath Kanoksirirath.
 All rights reserved under BSD 3-clause license.
"""

import numpy as np
import matplotlib.pyplot as plt

from IC import Initial_Cuxart2006 as defaultIC


def computeGradRi(model):
    "Return local, gradient Richardson number"

    dudz = model.grad_PtoS(model.u)[1:]
    dvdz = model.grad_PtoS(model.v)[1:]
    dUdz2 = np.maximum(np.power(dudz,2) + np.power(dvdz,2), 1e-12)
    dTdz = model.grad_PtoS(model.T)[1:]

    Ri = np.divide(model.para.g*dTdz/model.para.T_ref, dUdz2)

    # At surface layer level
    zrL = model.BC.rL*model.z_secondary[0]
    Ri = np.append(Ri, zrL)

    # Remove too strong Ri to plot
    Ri[Ri>5] = 5 # #####

    return Ri


def pltAdd():
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()

    return


def runQuasi(varName, nt, models, names, init=defaultIC(), style='-'):
    """
    Examine Quasi-state approximation of TKE in K-l model
    or Quasi-state approximation of Eps in K-eps models
    """

    font = {"size" : 14}
    plt.rc("font", **font)
    plt.rc('legend', fontsize=11)

    ### Running
    for i in range(len(models)) :

        print('Starting model', i+1,':', names[i])

        # Initialization
        if init != None :
            models[i].u = init.u(models[i])
            models[i].v = init.v(models[i])
            models[i].T = init.T(models[i])
            if hasattr(models[i], 'TKE') :
                models[i].TKE = init.TKE(models[i])
            else:
                print('This function is not for k-l or k-e models.')
                return
            if hasattr(models[i], 'Eps') :
                models[i].Eps = init.Eps(models[i])
            else:
                if varName == 'Eps' :
                    print('This option is for k-e models.')
                    return

        # Run
        l2 = np.zeros(nt)
        time = np.zeros(nt)
        for j in range(nt):
            models[i].run()
            time[j] = models[i].time()

            # L2 error norm
            if varName == 'TKE' :
                quasi = models[i].Quasi_TKE()
                var = models[i].TKE
            elif varName == 'Eps' :
                quasi = models[i].Quasi_Eps()[0]
                var = models[i].Eps
            else:
                print('Invalid option!!')
                return
            l2[j] = (np.sum(np.power(quasi - var, 2))/np.sum(var))**0.5

            # Print progress
            if j % 5000 == 0 :
                print('Running', names[i],':', j,'timesteps have passed.')

        # Plot
        if varName == 'TKE' :
            plt.figure(0)
            plt.plot(time*60, l2, style, label=names[i])
            plt.figure(1)
            p = plt.plot(models[i].TKE, models[i].z_secondary[:-1], style,
                     label=names[i])
            plt.plot(models[i].Quasi_TKE(), models[i].z_secondary[:-1],
                     color=p[0].get_color(), linestyle='dotted')
            #         label=names[i]+'-Quasi', color=p[0].get_color(), linestyle='dotted')
        if varName == 'Eps' :
            plt.figure(2)
            plt.plot(time*60, l2, style, label=names[i])
            plt.figure(3)
            p = plt.plot(models[i].Eps, models[i].z_secondary[:-1], style,
                     label=names[i])
            plt.plot(models[i].Quasi_Eps()[0], models[i].z_secondary[:-1],
                     color=p[0].get_color(), linestyle='dotted')
            #         label=names[i]+' (Quasi)', color=p[0].get_color(), linestyle='dotted')

    ### Plotting
    print('Plotting results')

    if varName == 'TKE' :
        plt.figure(0)
        plt.xlabel('Time (min)')
        plt.ylabel(r'$\ell_2$ error norm of quasi-TKE')
        #plt.ylim(0, 0.4)
        pltAdd()
        plt.savefig('./L2_TKE.png')
        plt.figure(1)
        plt.xlabel(r'$TKE \ (m^2/s^2)$')
        plt.ylabel('z (m)')
        plt.ylim(0, models[0]._Lz)
        pltAdd()
        plt.savefig('./QuasiTKE.png')

    if varName == 'Eps' :
        plt.figure(2)
        plt.xlabel('Time (min)')
        plt.ylabel(r'$\ell_2$ error norm of quasi-$\epsilon$')
        #plt.ylim(0, 0.02)
        pltAdd()
        plt.savefig('./L2_Eps.png')
        plt.figure(3)
        plt.xlabel(r'$\epsilon \ (m^2/s^3)$')
        plt.ylabel('z (m)')
        plt.ylim(0, models[0]._Lz)
        pltAdd()
        plt.savefig('./QuasiEps.png')

    return


def runModels(nt, models, names, init=defaultIC(), style='-'):

    font = {"size" : 14}
    plt.rc("font", **font)
    plt.rc('legend', fontsize=11)

    ### Running
    for i in range(len(models)) :

        print('Starting model', i+1,':', names[i])

        # Initialization
        if init != None :
            models[i].u = init.u(models[i])
            models[i].v = init.v(models[i])
            models[i].T = init.T(models[i])
            if hasattr(models[i], 'TKE') :
                models[i].TKE = init.TKE(models[i])
            if hasattr(models[i], 'Eps') :
                models[i].Eps = init.Eps(models[i])

        # Run
        save = np.zeros([nt, 6])
        time = np.zeros(nt)
        for j in range(nt):
            models[i].run()
            save[j,:] = models[i].BC._SurfaceStress(models[i])
            time[j] = models[i].time()

            if j % 5000 == 0 :
                print('Running', names[i],':', j,'timesteps have passed.')

        # Protected variables
        Km, Kh = models[i]._eddyViscosity()

        # Derive more variables
        U = np.sqrt(np.power(models[i].u, 2) + np.power(models[i].v, 2))
        Uw = np.sqrt(np.power(models[i].uw, 2) + np.power(models[i].vw, 2))
        Ri = computeGradRi(models[i])

        # Profiles
        plt.figure(0)
        plt.plot(models[i].u, models[i].z_primary, style, label=names[i])
        plt.figure(1)
        plt.plot(models[i].v, models[i].z_primary, style, label=names[i])
        plt.figure(2)
        plt.plot(U, models[i].z_primary, style, label=names[i])
        plt.figure(3)
        plt.plot(models[i].T, models[i].z_primary, style, label=names[i])
        plt.figure(4)
        plt.plot(models[i].uw, models[i].z_secondary, style, label=names[i])
        plt.figure(5)
        plt.plot(models[i].vw, models[i].z_secondary, style, label=names[i])
        plt.figure(6)
        plt.plot(Uw, models[i].z_secondary, style, label=names[i])
        plt.figure(7)
        plt.plot(models[i].wT, models[i].z_secondary, style, label=names[i])
        plt.figure(8)
        plt.plot(Km, models[i].z_secondary[1:-1], style, label=names[i])
        plt.figure(9)
        plt.plot(Kh, models[i].z_secondary[1:-1], style, label=names[i])
        plt.figure(10)
        plt.plot(Ri, models[i].z_secondary[:-1], style, label=names[i])

        # Timeseries
        for j in range(6):
            plt.figure(11+j)
            plt.plot(time*60, save[:,j], style, label=names[i])

        if hasattr(models[i], 'TKE'):
            plt.figure(17)
            plt.plot(models[i].TKE, models[i].z_secondary[:-1], style,
                     label=names[i])
        if hasattr(models[i], 'Eps'):
            plt.figure(18)
            plt.plot(models[i].Eps, models[i].z_secondary[:-1], style,
                     label=names[i])

        if hasattr(models[i], 'leps'):
            plt.figure(18)
            Eps = models[i]._deps*np.divide(np.power(models[i].TKE[1:], 1.5),
                                         models[i].leps[1:])
            plt.plot(Eps, models[i].z_secondary[1:-1], style,
                     label=names[i])
            plt.figure(19)
            plt.plot(models[i].leps, models[i].z_secondary[:-1], style,
                     label=names[i])


    ### Plotting
    print('Plotting results')

    Lz = models[0]._Lz

    # Profiles of Main variables
    plt.figure(0)
    plt.xlabel(r'$\overline{u} \ (m/s)$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./u.png')
    # ---
    plt.figure(1)
    plt.xlabel(r'$\overline{v} \ (m/s)$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./v.png')
    # ---
    plt.figure(2)
    plt.xlabel(r'$\overline{U} \ (m/s)$')
    plt.ylabel('z (m)')
    plt.xlim(0.,11.)
    #plt.xlim(0.,13.)
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./Speed.png')
    # ---
    plt.figure(3)
    plt.xlabel(r'$\overline{\theta} \ (K)$')
    plt.ylabel('z (m)')
    plt.xlim(262.5,268)
    #plt.xlim(258,274)
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./T.png')

    # Profiles of Fluxes
    plt.figure(4)
    plt.xlabel(r'$\overline{u^{\prime}w^{\prime}} \ (m^{2}s^{-2})$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./uw.png')
    # ---
    plt.figure(5)
    plt.xlabel(r'$\overline{v^{\prime}w^{\prime}} \ (m^{2}s^{-2})$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./vw.png')
    # ---
    plt.figure(6)
    plt.xlabel(r'$\sqrt{\overline{u^{\prime}w^{\prime}} + \overline{v^{\prime}w^{\prime}}} \ (m^{2}s^{-2})$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    plt.xlim(0.,0.14)
    pltAdd()
    plt.savefig('./SpeedW.png')
    # ---
    plt.figure(7)
    plt.xlabel(r'$\overline{w^{\prime} \theta^{\prime}} \ (m/s)$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    #plt.xlim(-0.03,0.)
    pltAdd()
    plt.savefig('./wT.png')
    # ---
    plt.figure(8)
    plt.xlabel(r'$K_m (m^{2}s^{-1})$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./Km.png')
    # ---
    plt.figure(9)
    plt.xlabel(r'$K_h \ (m^2s^{-1})$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./Kh.png')
    # ---
    plt.figure(10)
    plt.xlabel(r'$Ri$')
    plt.ylabel('z (m)')
    plt.ylim(0.,Lz)
    pltAdd()
    plt.savefig('./Ri.png')


    # Timeseries of  BC
    plt.figure(11)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$\overline{u^{\prime}w^{\prime}}_0 \ (m^{2}s^{-2})$')
    #plt.ylim(-1,0)
    pltAdd()
    plt.savefig('./uw0.png')
    # ---
    plt.figure(12)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$\overline{v^{\prime}w^{\prime}}_0 \ (m^{2}s^{-2})$')
    pltAdd()
    plt.savefig('./vw0.png')
    # ---
    plt.figure(13)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$\overline{w^{\prime}\theta^{\prime}}_0 \ (Km/s)$')
    pltAdd()
    plt.savefig('./wT0.png')
    # ---
    plt.figure(14)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$1/L \ (m^{-1})$')
    pltAdd()
    plt.savefig('./rL.png')
    # ---
    plt.figure(15)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$u_* \ (m/s)$')
    plt.ylim(0.2,0.5)
    #plt.xlim(0,540)
    pltAdd()
    plt.savefig('./U_star.png')
    # ---
    plt.figure(16)
    plt.xlabel('Time (min)')
    plt.ylabel(r'$\theta_* \ (K)$')
    pltAdd()
    plt.savefig('./T_star.png')
    # ---
    plt.figure(17)
    plt.xlabel(r'$TKE \ (m^2/s^2)$')
    plt.ylabel('z (m)')
    pltAdd()
    plt.savefig('./TKE.png')
    # ---
    plt.figure(18)
    plt.xlabel(r'$\epsilon \ (m^2/s^3)$')
    plt.ylabel('z (m)')
    pltAdd()
    plt.savefig('./Eps.png')

    plt.figure(19)
    plt.xlabel(r'$l_\epsilon \ (m)$')
    plt.ylabel('z (m)')
    pltAdd()
    plt.savefig('./leps.png')

    return
