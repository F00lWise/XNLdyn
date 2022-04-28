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

    pulse_energy_J = 1e-20 # J

    PAR.I0 =  [XNLdyn.photons_per_J(PAR.E_i[0])*pulse_energy_J,]

    sim = XNLdyn.XNLsim(PAR, DEBUG=True)

    print(PAR.enax_j)
    #

    #sim_options = dict(t_span=[-40, 60], method='RK45', rtol=1e-6, atol=1e-8, plot=True, return_full_solution=True)

    #incident, transmitted, sol = sim.run(**sim_options)

    #print('Transmission: ', 100 * transmitted / incident, ' %')


    #plt.show()