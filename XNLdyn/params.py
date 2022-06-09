Nsteps_z = 40      # Steps in Z
N_photens= 1      # Number of distinct incident resonant photon energies E_i

timestep_min = 0.5 #fs
## Sample data
Z = 20 #Sample thickness in nm
atomic_density = 91.4  # atoms per nm³
photon_bandwidth = 0.6 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV
temperature = 300      # Kelvin

# Electronic state numbers per atom
core_states = 2
valence_GS_occupation = 10

DoS_shapefile = './DoSdata.npy'
DoS_band_origin = -10 #eV
DoS_band_dd_end = 3. 

## Rates and cross sections

tau_CH = 11.42         # Core hole lifetime / fs
tau_th = 10    # Redistribution time of electrons in the VB / fs
tau_free = 40e20         # Free electron lifetime
lambda_res_Ei = (3,)  # Absorptions length of resonant photon energies / nm
lambda_nonres = 1e20    # Absorption length of non-resonant photons, assumed equal for all / nm

## Fermi Energy
E_f = 850.7 #eV

N_j = 70      # Number of points E_j with which the valence system is resolved
Energy_axis_max = 600
Energy_axis_fine_until = 20
Energy_axis_min = -10

## Incident photon profile
I0       = [3,]   # Pulse energy density in photons per nm²
t0       = [0,]   # Arrival time on time-axis / fs
tdur_sig = [3,]   # Rms pulse duration / fs
E_i      = [858,] # Photon Energies of incident pulses / eV