import matplotlib.pyplot as plt
import numpy as np


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

def plot_results(PAR, sol, sol_photon_densities):
    ## Plotting
    soly = sol.y.reshape((PAR.Nsteps_z, PAR.states_per_voxel, len(sol.t)))
    sol_core = soly[:, 0, :]
    sol_free = soly[:, 1, :]
    sol_VB = soly[:, 2, :]
    sol_T = soly[:, 3, :]
    sol_Efree = soly[:, 4, :]
    sol_Ej = soly[:, 5:, :]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    plt.sca(axes[0,0])
    plt.title('State occupation changes')
    plt.plot(sol.t, 1- (np.mean(sol_core, 0) / PAR.M_CE), label='Core holes')
    plt.plot(sol.t, 1- (sol_core[0] / PAR.M_CE) , label='Core holes [0]')

    plt.plot(sol.t, (np.mean(sol_VB, 0) - PAR.rho_VB_0) / PAR.M_VB, label='Valence band occupation')
    plt.plot(sol.t, (sol_VB[0] - PAR.rho_VB_0) / PAR.M_VB, label='Valence @surface')

    plt.plot(sol.t, np.mean(sol_Ej, 0).T / PAR.M_Ej,
             label=[f'{PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])

    plt.ylabel('Occupation')
    plt.xlabel('t (fs)')
    plt.legend()

    plt.sca(axes[0,1])
    plt.title('Kinetic electrons')

    plt.plot(sol.t, np.mean(sol_free, 0), label='Kinetic electrons')
    plt.plot(sol.t, sol_free[0], label='Surface')

    plt.ylabel('Occupation')
    plt.xlabel('t (fs)')
    plt.legend()

    plt.sca(axes[1,0])
    plt.title('Energies Averaged over z')
    plt.plot(sol.t, np.mean(sol_T, 0), label='Valence Thermal Energy')
    plt.plot(sol.t, np.mean(sol_Efree, 0), label='Free Electron Energy')
    plt.plot(sol.t, sol_T[0], label='Valence @ Surface')
    plt.plot(sol.t, sol_Efree[0], label='Free surface')
    plt.ylabel('E (eV)')
    plt.xlabel('t (fs)')
    plt.legend()

    plt.sca(axes[1,1])
    plt.title('Photons')
    plt.xlabel('t (fs)')

    plt.plot(sol.t, sol_photon_densities[-1, :, :].T / sol_photon_densities[0, :, :].T,
             label=[f'{PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])
    plt.legend()
    plt.ylabel('T')
    plt.xlabel('t (fs)')


    plt.tight_layout()
    plt.show()