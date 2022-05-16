Nsteps_z = 10     # Steps in Z
N_photens= 1      # Number of distinct incident resonant photon energies E_i

timestep_min = 1 #fs
## Sample data
Z = 10 #Sample thickness in nm
atomic_density = 91.4  # atoms per nm³
photon_bandwidth = 0.4 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV
temperature = 300      # Kelvin

# Electronic state numbers per atom
core_states = 2
#total_valence_states = 12
valence_GS_occupation = 10

DoS_shapefile = './DoSdata.npy'
#work_function = -5.15 # eV
DoS_band_origin = -10 #eV
DoS_band_dd_end = 5.15

## Rates and cross sections

tau_CH = 10            # Core hole lifetime / fs
tau_th = 3             # Redistribution time of electrons in the VB / fs
tau_free = 10          # Free electron lifetime
lambda_res_Ei = (10,)  # Absorptions length of resonant photon energies / nm
lambda_nonres = 190    # Absorption length of non-resonant photons, assumed equal for all / nm


## Fermi Energy
E_f = 850.7 #eV

N_j = 50          # Number of points E_j with which the valence system is resolved
Energy_axis_max = 80
Energy_axis_fine_until = 10
Energy_axis_min = -10

## Incident photon profile
I0       = [3,]   # Pulse energy density in photons per nm²
t0       = [0,]   # Arrival time on time-axis / fs
tdur_sig = [2,]  # Rms pulse duration / fs
E_i      = [858,] # Photon Energies of incident pulses / eV