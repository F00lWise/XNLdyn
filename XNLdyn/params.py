Nsteps_z = 50     # Steps in Z
N_photens= 1      # Number of distinct incident resonant photon energies E_i

timestep_min = 1.3 #fs
## Sample data
Z = 25 #Sample thickness in nm
atomic_density = 91.4  # atoms per nm³
photon_bandwidth = 0.8 # The energy span of the valence band that each resonant energy interacts with. Flat-top. / eV
temperature = 300      # Kelvin

# Electronic state numbers per atom
core_states = 2
valence_GS_occupation = 10

DoS_shapefile = './DoSdata.npy'
#work_function = -5.15 # eV
DoS_band_origin = -10 #eV
DoS_band_dd_end = 3.15

## Rates and cross sections

tau_CH = 4           # Core hole lifetime / fs
tau_th = 2            # Redistribution time of electrons in the VB / fs
tau_free = 20          # Free electron lifetime
lambda_res_Ei = (10,)  # Absorptions length of resonant photon energies / nm
lambda_nonres = 190    # Absorption length of non-resonant photons, assumed equal for all / nm

## Fermi Energy
E_f = 850.7 #eV

# Size of the lookup tables for connecting inner energy and population to chemical potential and temperature
lookup_table_data =  {'size': 200,
                     'chem_pot_minstep': 0.05,
                     'chem_pot_min': -550,
                     'chem_pot_max': 25,
                     'T_max': 250/8.617333262145e-5}

N_j = 100          # Number of points E_j with which the valence system is resolved
Energy_axis_max = 100
Energy_axis_fine_until = 30
Energy_axis_min = -10

## Incident photon profile
I0       = [3,]   # Pulse energy density in photons per nm²
t0       = [0,]   # Arrival time on time-axis / fs
tdur_sig = [4,]  # Rms pulse duration / fs
E_i      = [858,] # Photon Energies of incident pulses / eV