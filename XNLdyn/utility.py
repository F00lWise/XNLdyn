import matplotlib.pyplot as plt
import numpy as np
import XNLdyn

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
    #print('Incident: ', incident)
    #print('Transmitted: ', transmitted)
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
    """
    :param Nsteps_r: Number
    :param pulse_energy_max: Joule
    :param pulse_profile_sigma: pulse_profile_sigma
    """
    r_centers, dr, dA = make_integration_axis(Nsteps_r, pulse_profile_sigma)
    
    fluences_phot_nm2 = pulse_energy_max * spotgauss(r_centers, pulse_profile_sigma)
    return fluences_phot_nm2, dA

def photons_per_J(photon_energy):
    echarge = 1.602176634e-19 #J/eV
    return 1/(photon_energy*echarge)