import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sc

# Multiprocessing
import multiprocessing as mp
from multiprocessing import Pool

# For the progress bar
from ipywidgets import IntProgress
from IPython.display import display

# This package
import XNLdyn

import warnings
warnings.filterwarnings('default')

from datetime import datetime
import os
import pickle

if __name__ == '__main__':
    
    #### Prepare for saving results
    save_parent_dir = './simulation_results/'
    save_dirname = datetime.now().strftime("%d-%m-%Y_%H-%M/")
    save_path = os.path.join(save_parent_dir,save_dirname)
    # Make directory
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print('Warning! Writing into existing directory!')
        
    
    use_multiprocessing = True

    ## Set up the problem
    PAR = XNLdyn.XNLpars()


    N_local_fluences_to_calculate = 30
    N_pulse_energies = 20
    Nsteps_r = 100
    pulse_energy_max = 0.3e-6 # J
    pulse_profile_sigma= 400 # nm rms


    ### Configure which fluences to calculate
    fluences_joules_nm2, dA = XNLdyn.calculate_fluences(Nsteps_r, pulse_energy_max, pulse_profile_sigma) 

    fluences_simulated_joules_nm2 = np.logspace(np.log10(np.min(fluences_joules_nm2*1e-2)),
                                     np.log10(np.max(fluences_joules_nm2)),
                                     N_local_fluences_to_calculate)

    fluences_simulated_joules_cm2 = fluences_simulated_joules_nm2 * 1e14
    fluences_simulated_photons_nm2_rough = fluences_simulated_joules_nm2 * XNLdyn.photons_per_J(850)

    ### Load spectrum
    N_points_E = 42# 23
    Erange = 3
    aufloesung = 0.27

    load_spectrum_file = '/home/engelr/Beamtime_XFEL2018/Eval4/NL-Spectra_runs_[141, 148, 151, 158, 164]_base.npy'

    ld = np.load(load_spectrum_file,allow_pickle=True).item()
    E_fermi = 850.7
    spec = ld['Spectrum0'][10:-10]
    spec_sm = ld['Spectrum_sm0'][10:-10]
    enax = ld['Enax'][10:-10]-1.5        # Energy calibration difference between Fermi edges
    OD_nonres = spec_sm[0]
    A_spec_sm = 10**(spec_sm-OD_nonres)
    lambda_nonres = PAR.Z / np.log(10**OD_nonres)
    spec_lambda_sm= PAR.Z / np.log(A_spec_sm)
    E_min, E_max = E_fermi-1.5*Erange, E_fermi+Erange
    enax_abs = np.linspace(E_min, E_max+2, N_points_E) # Absolute energy axis to sample
    enax_rel = enax_abs-E_fermi # Relative energy axis to Fermi Energy, i.e. detuning
    dE = enax_abs[1]-enax_abs[0]
    pendepths_res = np.interp(enax_abs, enax,spec_lambda_sm)
    pendepths_res_symm = pendepths_res.copy()
    pendepths_res_symm[enax_rel<0] = np.interp((E_fermi-enax_abs[enax_rel<0]), enax-E_fermi,spec_lambda_sm)
    
    ### Run Simulation
    
    print(f"Starting Simulation on {int(os.environ['SLURM_CPUS_ON_NODE'])} processes")
    
    sim_options = dict(t_span=[-42, 42],method='RK45', rtol=1e-4, atol=1e-8, plot = False)
    
    timeout = 1200

    try:
        mp.set_start_method('fork')  # 'spawn' on windows, "fork" or "forkserver" on unix machines
    except RuntimeError:
        print('Cannot set start method - maybe already set?')
        pass
    with Pool(processes=int(os.environ['SLURM_CPUS_ON_NODE'])) as pool:
        tasklist = {}
        for photon_energy, pendepdth in zip(enax_abs, pendepths_res_symm):
            for fluence in fluences_simulated_joules_nm2:
                #progressbar.value += 1
                fluence_photons = fluence * XNLdyn.photons_per_J(photon_energy)
                tasklist[(photon_energy,fluence)]=pool.apply_async(XNLdyn.run_modified_simulation,\
                                                                   (*(PAR, sim_options,\
                                                                      ['I0_i','E_i_abs','lambda_res_Ei','lambda_nonres'],\
                                                                      [(fluence_photons,),(photon_energy,),(pendepdth,),lambda_nonres]),\
                                                                   ))
        
        resultdict = {}
        for key in tasklist:
            resultdict[key] = tasklist[key].get(timeout=timeout)
            
            
    ### Process results
    fl_dep_spectrum_I = np.zeros((N_points_E,N_local_fluences_to_calculate))
    fl_dep_spectrum_T = np.zeros((N_points_E,N_local_fluences_to_calculate))

    for i_photen in range(N_points_E):
        for i_pulseen in range(N_local_fluences_to_calculate):
            fl_dep_spectrum_I[i_photen, i_pulseen], fl_dep_spectrum_T[i_photen, i_pulseen]  = \
                resultdict[(enax_abs[i_photen],fluences_simulated_joules_nm2[i_pulseen])]

        
    T = fl_dep_spectrum_T/fl_dep_spectrum_I
    
    # z-stacks plot
    plotcols =  mpl.cm.plasma(np.linspace(.0,0.9,N_local_fluences_to_calculate))#YlOrRd_r
    plt.rcParams.update({'font.size': 9})
    fig = plt.figure(figsize =(3.5,3.))

    dE = np.mean(enax_rel[1:]-enax_rel[:-1])

    for i in range(N_local_fluences_to_calculate)[:]:
        if np.mod(i,3)==0:
            spec = 1e3*np.log10(1/T[:,i])
            spec_sm = sc.ndimage.gaussian_filter(spec,aufloesung/dE)
            #plt.plot(enax_abs,spec, color = plotcols[i], lw=0.5)
            fluence_J = fluences_simulated_joules_nm2[i]*1e14
            lab = f'{fluence_J:.3f} J/cm²' if fluence_J >1 else f'{fluence_J*1e3:.3f} mJ/cm²' 
            plt.plot(enax_abs,spec_sm, color = plotcols[i], label = lab)


    plt.plot(ld['Enax'][10:-10]-1.5, 1e3*ld['Spectrum_sm0'][10:-10], label = 'Input spectrum')
    plt.legend(fontsize = 7)

    #plt.title('Results for homogeneous illumination')
    plt.ylabel('optical density (mOD)')
    plt.xlabel('photon energy / eV')
    plt.tight_layout()
    plt.xlim(847, 853.5)

    plt.savefig('./plots/homogeneous_spectra.png', dpi = 600)
    plt.savefig(save_path+'homogeneous_spectra.png')

   # These are the pulse energies for which we evaluate stuff
    final_pulse_energies = np.logspace(np.log10(1e-4/N_pulse_energies), np.log10(1), N_pulse_energies)* pulse_energy_max

    final_transmissions = np.zeros((N_points_E, N_pulse_energies))
    final_incidence_check = np.zeros((N_points_E, N_pulse_energies))
    final_incidence_peaks = np.zeros(N_pulse_energies)
    for i_photen in range(N_points_E):

        for ipe, pulse_en in enumerate(final_pulse_energies):
            local_fluences, dA = XNLdyn.calculate_fluences(Nsteps_r, pulse_en, pulse_profile_sigma)


            local_transmitted = np.interp(local_fluences, fluences_simulated_joules_nm2,
                                          fl_dep_spectrum_T[i_photen,:])# tr[:, 0]
            final_transmissions[i_photen,ipe] = np.sum(local_transmitted*dA)
            local_incidence_check = np.interp(local_fluences, fluences_simulated_joules_nm2, fl_dep_spectrum_I[i_photen,:])
            final_incidence_check[i_photen,ipe] = np.sum(local_incidence_check*dA) # should result equal final_pulse_energies
            final_incidence_peaks[ipe] = np.max(local_incidence_check) 

    ### Final figure
    fig = plt.figure(figsize =(3.5,3.))
    plotcols =  mpl.cm.plasma(np.linspace(.0,0.9,N_pulse_energies))#YlOrRd_r

    for i in range(N_pulse_energies)[::]:
        spec = 1e3*np.log10(final_incidence_check[:,i]/final_transmissions[:,i])
        spec_sm = sc.ndimage.gaussian_filter(spec,aufloesung/dE)
        plt.plot(enax_abs, spec, color = plotcols[i], lw=0.5)#, label ='For one z-stack'

        #plt.plot(enax_abs, spec_sm, color = plotcols[i], label = f'{final_pulse_energies[i]*1e6:.2f} µJ')#, label ='For one z-stack'
        fluence_J = final_pulse_energies[i]
        #{fluence_J*1e9:.3f} nJ,  
        lab =  f'<{1e14*final_incidence_peaks[i]/XNLdyn.photons_per_J(PAR.E_i_abs[0]):.3f} J/cm²' 
        #plt.plot(enax_abs, spec_sm, color = plotcols[i], label =lab)

    plt.plot(ld['Enax'][10:-10]-1.5, 1e3*ld['Spectrum_sm0'][10:-10], label = 'Input spectrum')
    plt.legend(fontsize = 7)
    plt.xlabel('photon energy (eV)')
    plt.ylabel('optical density (mOD)')
    #plt.title('Spot reconstruction')
    plt.tight_layout()
    plt.xlim(847, 853.5)
    plt.savefig('./plots/final_spectra.png', dpi = 600)

    plt.savefig(save_path+'final_spectra.png')

    print('done')