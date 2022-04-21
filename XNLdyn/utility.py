import matplotlib.pyplot as plt
import numpy as np



def spotgauss(r, sigma):
    """
    Returns a centered Gaussian distribution normalized to the area
    (in polar coordinates r and phi, but due to angular symmetry, phi is not required)
    :param r:    points in radius (vector)
    :param sigma: sigma of Gaussian
    :return: normalized amplitude
    """
    return 1/sigma**2 * np.exp(-(r**2)/(2*sigma**2))

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