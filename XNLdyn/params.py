import numpy as np

Nsteps_z = 100     # Steps in Z
N_photens= 3      # Number of distict resonant photon energies


## Sample data
Z = 25 #Sample thickness in nm
atomic_density = 91.4 # atoms per nm³
photon_bandwidth = 0.3 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV
temperature = 300 # Kelvin

# Electronic state numbers per atom
core_states = 2
total_valence_states = 10

DoS_shapefile = './DoSdata.npy'

## Rates and cross sections

tau_CH = 4       # Core hole lifetime / fs
tau_free = 10*0.8   # Kinetic electron lifetime / fs
tau_therm = 10*0.1  # Redistribution time of electrons in the VB / fs
lambda_res_Ej = (10,11,9)  # Absorptions length of resonant photon energies / nm
lambda_nonres = 500    # Absorption length of non-resonant photons, assumed equal for all / nm


## Fermi Energy
E_f = 850 #eV

## Incident photon profile
I0       = [7,10,8]   # Pulse energy density in photons per nm²
t0       = [0,5,-5]   # Arrival time on time-axis / fs
tdur_sig = [7,8,5]   # Rms pulse duration / fs
E_j      = [849,852,846] # Photon Energies of incident pulses / eV