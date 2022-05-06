Nsteps_z = 20     # Steps in Z
N_photens= 2      # Number of distinct incident resonant photon energies E_i
N_j = 50          # Number of points E_j with which the valence system is resolved

## Sample data
Z = 20 #Sample thickness in nm
atomic_density = 91.4 # atoms per nm³
photon_bandwidth = 0.3 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV
temperature = 300 # Kelvin

# Electronic state numbers per atom
core_states = 2
total_valence_states = 12
valence_GS_occupation = 10

DoS_shapefile = './DoSdata.npy'

## Rates and cross sections

tau_CH = 30       # Core hole lifetime / fs
tau_th = 5     # Redistribution time of electrons in the VB / fs
tau_free = 10   # Free electron lifetime
lambda_res_Ei = (10, 5)  # Absorptions length of resonant photon energies / nm
lambda_nonres = 500    # Absorption length of non-resonant photons, assumed equal for all / nm


## Fermi Energy
E_f = 850.7 #eV

## Incident photon profile
I0       = [3,2]   # Pulse energy density in photons per nm²
t0       = [0,10]   # Arrival time on time-axis / fs
tdur_sig = [10,10]  # Rms pulse duration / fs
E_i      = [858,847] # Photon Energies of incident pulses / eV