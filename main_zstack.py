import numpy as np
import XNLdyn
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool



if __name__ == '__main__':

    ## Set up the problem
    PAR = XNLdyn.XNLpars()

    N_local_fluences_to_calculate = 30
    N_pulse_energies = 20
    Nsteps_r = 100

    pulse_energy_J = 3e-15 # J

    PAR.I0_i = [XNLdyn.photons_per_J(PAR.E_i_abs[0])*pulse_energy_J]
    print('Photon numbers per atom for this simulation: ', np.array(PAR.I0_i)/PAR.atomic_density)

    sim = XNLdyn.XNLsim(PAR, DEBUG=True)#, load_tables=False

    #PAR.FermiSolver.plot_lookup_tables()

    sim_options = dict(t_span=[-35, 35], method='RK45', rtol=1e-4, atol=1e-8, plot=True, return_full_solution=True)

    incident, transmitted, sol = sim.run(**sim_options)

    print('Transmission: ', 100 * transmitted / incident, ' %')


    plt.show(block = True)