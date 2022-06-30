# Axis information
Nsteps_z = 20     # Steps in Z


N_j = 70      # Number of points E_j with which the valence system is resolved
Energy_axis_max = 600
Energy_axis_fine_until = 20
Energy_axis_min = -10

timestep_min = 0.7 #fs


## Sample data
Z = 20                 #Sample thickness in nm
atomic_density = 91.4  # atoms per nm³
temperature = 300      # Kelvin

# Electronic state numbers per atom
core_states = 4
valence_GS_occupation = 10

## Fermi Energy
E_f = 850.7 #eV
photon_bandwidth = 0.3 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV

# DoS info
DoS_shapefile = './DoSdata.npy'
DoS_band_origin = -10 #eV
DoS_band_dd_end = 3. 

## Rates and cross sections
tau_CH = 1.37          # Core hole lifetime / fs
tau_th = 10             # Redistribution time of electrons in the VB / fs
tau_free = 30           # Free electron lifetime
lambda_nonres = 1e20   # Absorption length of non-resonant photons, assumed equal for all / nm

## Incident photon profile
N_photens= 1      # Number of distinct incident resonant photon energies E_i
lambda_res_Ei = (1e20,)  # Absorptions length of resonant photon energies / nm
I0       = [3,]   # Pulse energy density in photons per nm²
t0       = [0,]   # Arrival time on time-axis / fs
tdur_sig = [13,]   # Rms pulse duration / fs
E_i      = [858,] # Photon Energies of incident pulses / eV