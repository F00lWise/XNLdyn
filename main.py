import numpy as np
import matplotlib.pyplot as plt
from XNLdyn import XNLsim
from scipy.integrate import solve_ivp

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## Set up the problem
    PAR = XNLsim.XNLpars()
    sim = XNLsim.XNLsim(PAR)



    ### Solve Main problem
    sol = solve_ivp(sim.time_derivative, t_span=[-20, 150], \
                    dense_output=True, y0=PAR.state_vector_0.flatten(), method='DOP853', rtol=1e-3, atol=1e-8)

    soly = sol.y.reshape((PAR.Nsteps_z, PAR.states_per_voxel, len(sol.t)))


    ### Since they weren't saved, calculate transmission again
    sol_photon_densities = np.zeros((PAR.Nsteps_z, PAR.N_photens, len(sol.t)))
    for it, t in enumerate(sol.t):
        sol_photon_densities[:, :, it] = sim.z_dependence(t, soly[:, :, it])




    ## Plotting
    soly = sol.y.reshape((PAR.Nsteps_z, PAR.states_per_voxel, len(sol.t)))
    sol_core = soly[:, 0, :]
    sol_free = soly[:, 1, :]
    sol_VB = soly[:, 2, :]
    sol_T = soly[:, 3, :]
    sol_Efree = soly[:, 4, :]
    sol_Ej = soly[:, 5:, :]

    fig, axes = plt.subplots(3, 1, figsize=(4, 7))
    plt.sca(axes[0])
    plt.title('States Averaged over z')
    plt.plot(sol.t, np.mean(sol_core, 0) / PAR.M_CE, label='Core occupation')
    plt.plot(sol.t, sol_core[0] / PAR.M_CE, label='Core occupation [0]')
    plt.plot(sol.t, np.mean(sol_free, 0), label='Kinetic electrons')
    plt.plot(sol.t, np.mean(sol_VB, 0) / PAR.M_VB, label='Valence band occupation')

    plt.plot(sol.t, np.mean(sol_Ej, 0).T / PAR.M_Ej,
             label=[f'{PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])
    plt.ylabel('Occupation')
    plt.xlabel('t (fs)')
    plt.legend()

    plt.sca(axes[1])
    plt.title('Energies Averaged over z')
    plt.plot(sol.t, np.mean(sol_T, 0), label='Valence Thermal Energy')
    plt.plot(sol.t, np.mean(sol_Efree, 0), label='Free Electron Energy')
    plt.ylabel('E (eV)')
    plt.xlabel('t (fs)')
    plt.legend()

    plt.sca(axes[2])
    plt.title('Photons')
    plt.xlabel('t (fs)')

    plt.plot(sol.t, sol_photon_densities[-1, :, :].T / sol_photon_densities[0, :, :].T,
             label=[f'(x10) {PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])
    plt.legend()
    plt.ylabel('T')
    plt.xlabel('t (fs)')
    plt.tight_layout()
