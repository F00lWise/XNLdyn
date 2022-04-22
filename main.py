import numpy as np
import XNLdyn
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool

def run_modified_simulation(PAR, sim_options, changed_parameters, new_values):
    print(f'Initializing a simulation where {changed_parameters} are changed to {new_values}\n')
    # Change parameters in PAR
    assert len(changed_parameters)==len(new_values)
    for i, par in enumerate(changed_parameters):
        setattr(PAR, par, new_values[i])

    # Update the simulation and calculate derived parameters
    sim = XNLdyn.XNLsim(PAR)

    # run it!
    incident, transmitted = sim.run(**sim_options)
    print('Incident: ', incident)
    print('Transmitted: ', transmitted)
    print('Transmission: ', 100 * transmitted/incident, ' %')
    return incident, transmitted

def make_integration_axis(Nsteps_r, sig):
    """
    Prepares the radial axis to integrate a radially symmetric spot.
    The number of steps <Nsteps_r> is distributed in irregular intervals to improve the sampling of a Gaussian spot.
    Especially relevant are the returns r_centers at which the simulation will run and the area dA for which that spot is valid.

    :param Nsteps_r: Number of points at which the simulation will run
    :param sig:  Sigma of the Gaussian
    :return: r_edges, r_centers, dr, dA
    """
    Nsteps_r +=1 # Evaluation points to edge-points
    Nsteps_r_inner = int(np.floor(Nsteps_r / 12))  # This many steps between 0 and sigma/4
    Nsteps_r_center = int(
        3 * np.floor(Nsteps_r / 4))  # This many between sigma/4 and 2.5*sigma (most relevant, highest derivative)
    Nsteps_r_outer = Nsteps_r - Nsteps_r_inner - Nsteps_r_center  # This many in the outer remaining region
    R = 5 * sig
    r_edges = np.concatenate((np.linspace(0, sig / 4, Nsteps_r_inner, endpoint=False),
                              np.linspace(sig / 4, 2.5 * sig, Nsteps_r_center, endpoint=False),
                              np.linspace(2.5 * sig, R, Nsteps_r_outer)))
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    dr = r_edges[1:] - r_edges[:-1]
    dA = r_centers * dr
    return r_centers, dr, dA

def spotgauss(r, sigma):
    """
    Returns a centered Gaussian distribution normalized to the area
    (in polar coordinates r and phi, but due to angular symmetry, phi is not required)
    :param r:    points in radius (vector)
    :param sigma: sigma of Gaussian
    :return: normalized amplitude
    """
    return 1/sigma**2 * np.exp(-(r**2)/(2*sigma**2))

def calculate_fluences(Nsteps_r, pulse_energy_max, pulse_profile_sigma):

    r_centers, dr, dA = make_integration_axis(Nsteps_r, pulse_profile_sigma)
    fluences_J_um2 = pulse_energy_max * spotgauss(r_centers, pulse_profile_sigma)*dA
    fluences_J_nm2 = fluences_J_um2/1e6
    fluences_eV_nm2 = fluences_J_nm2/PAR.echarge
    fluences_photons_nm2 = fluences_eV_nm2/850 # change to par_Ej when I make this vectorized

    return fluences_photons_nm2, dA

if __name__ == '__main__':

    use_multiprocessing = True

    ## Set up the problem
    PAR = XNLdyn.XNLpars()


    N_local_fluences_to_calculate = 30
    N_pulse_energies = 20
    Nsteps_r = 100

    pulse_energy_max = 10e-6 # J
    pulse_profile_sigma= 2 # µm rms

    fluences_photons_nm2, _ = calculate_fluences(Nsteps_r, pulse_energy_max, pulse_profile_sigma) # These are just to get an idea where to calculate Fluences
    fluences_simulated = np.logspace(np.log10(np.min(fluences_photons_nm2)),
                                     np.log10(np.max(fluences_photons_nm2)),
                                     N_local_fluences_to_calculate)


    sim_options = dict(t_span=[-40, 60],method='RK45', rtol=1e-6, atol=1e-8, plot = False)



    if use_multiprocessing:
        mp.set_start_method('spawn')  # may try "fork" or "forkserver" on unix machines
        with Pool(processes=10) as pool:
            tasklist = []
            """for fluence in fluences_simulated:
                tasklist.append(
                    pool.apply_async(run_modified_simulation,
                                     (*(PAR, sim_options, ['I0', ], [fluence, ]),)
                                     )
                )
            """
            tasklist = [pool.apply_async(run_modified_simulation,(*(PAR, sim_options,  ['I0',] , [fluence,]),)
                                                 ) for fluence in fluences_simulated]
            resultlist = [res.get(timeout=60) for res in tasklist]
    else:
        resultlist = []
        for fluence in fluences_simulated:
            result = run_modified_simulation(PAR, sim_options, changed_parameters = ['I0',], new_values = [fluence,])
            resultlist.append(result)
    print(resultlist)

    inc = np.array([r[0] for r in resultlist])
    tr = np.array([r[1] for r in resultlist])
    T = tr/inc

    plt.figure()
    plt.plot(fluences_simulated, T, '.-', label ='For one z-stack')
    plt.xlabel('Fluence (photons/nm2)')
    plt.ylabel('Transmission')
    plt.legend(loc = 'lower right')

    final_transmissions = np.zeros(N_pulse_energies)
    final_incidence_check = np.zeros(N_pulse_energies)
    final_consintency_check = np.zeros(N_pulse_energies)

    final_pulse_energies = np.linspace(1/N_pulse_energies, 1, N_pulse_energies)* pulse_energy_max
    for ipe, pulse_en in enumerate(final_pulse_energies):
        local_fluences, dA = calculate_fluences(Nsteps_r, pulse_en,
                                                  pulse_profile_sigma)

        local_transmitted = np.interp(local_fluences, fluences_simulated, tr[:, 0])
        local_incidence_check = np.interp(local_fluences, fluences_simulated, inc[:, 0])
        local_consistency_check = np.interp(local_fluences, fluences_simulated, fluences_simulated)

        final_transmissions[ipe] = np.sum(local_transmitted*dA)
        final_incidence_check[ipe] = np.sum(local_incidence_check*dA) # should result equal final_pulse_energies
        final_consintency_check[ipe] = np.sum(local_consistency_check*dA) # should result equal final_pulse_energies

    plt.figure()
    plt.plot(final_pulse_energies, final_transmissions*850*1e6*PAR.echarge, 'C1.-', label = 'Transmitted')
    plt.plot(final_pulse_energies, final_consintency_check*850*1e6*PAR.echarge, 'C2x-', label = 'Incident reproduced')
    plt.plot(final_pulse_energies, final_incidence_check*850*1e6*PAR.echarge, 'C3.-', label = 'Incidences reproduced for reference')
    plt.xlabel('Pulse Energy (J)')
    plt.ylabel('Transmitted photons')
    plt.legend(loc = 'lower left')

    plt.show(block = True)

    print('done')