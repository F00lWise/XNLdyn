import numpy as np


# Import all the parameters defined in the params file and processed in process_params
from .params import *

class XNLpars:
    def __init__(self):
        # Some constants
        self.kB = 8.617333262145e-5 # Boltzmann Konstant / eV/K
        self.lightspeed = 299792458 # m/s
        self.hbar = 6.582119569e-15 # eV s
        self.echarge = 1.602176634e-19 # J/eV
        ## Here I just "package" the variables from the params file into class attributes
        
        self.Nsteps_z = Nsteps_z     # Steps in Z
        self.N_photens= N_photens     # Number of distict resonant photon energies
        ## Sample thickness
        self.Z = Z #nm

        self.atomic_density = atomic_density  # atoms per nm³
        self.photon_bandwidth = photon_bandwidth # The energy span of the valence band that each resonant energy interacts with./ eV
        self.temperature = temperature # Kelvin

        # Electronic state numbers per atom
        self.core_states = core_states 
        self.total_valence_states = total_valence_states  
        self.DoS_shapefile = DoS_shapefile

        ## Rates and cross sections
        self.tau_CH = tau_CH
        self.tau_free = tau_free
        self.tau_therm = tau_therm
        self.lambda_res_Ej = lambda_res_Ej
        self.lambda_nonres = lambda_nonres
        
        ## Fermi Energy
        self.E_f =  E_f
        
        ## Incident photon profile       
        self.I0        = np.array(I0)
        self.t0        = np.array(t0)
        self.tdur_sig  = np.array(tdur_sig)
        self.E_j       = np.array(E_j)

        assert (N_photens == len(I0) == len(t0) == len(tdur_sig) == len(E_j) == len(lambda_res_Ej) ),\
            'Make sure all photon pulses get all parameters!'
        
        
    def make_derived_params(self, sim):
        ## now some derived quantities
        self.zstepsize = self.Z/self.Nsteps_z
        self.zaxis = np.arange(0,self.Z,self.zstepsize)

        self.states_per_voxel = 5+self.N_photens # Just the number of entries for later convenience

        # Load DoD data
        ld = np.load(self.DoS_shapefile)
        self.DoSdata = {}
        self.DoSdata['x'] = ld[:,0]
        self.DoSdata['y'] = ld[:,1]
        self.DoSdata['osi'] = np.array([np.trapz(self.DoSdata['y'][:i], x=self.DoSdata['x'][:i]) for i in range(len(self.DoSdata['x']))]) # one-sided integral for calculating Fermi energy

        ## Multiplicities - these are global for all t and z
        self.M_CE = self.atomic_density * self.core_states
        self.M_VB = self.atomic_density * self.total_valence_states
        self.M_Ej = np.array([self.M_VB * self.get_resonant_states(self.E_j[j]-self.E_f, self.photon_bandwidth)\
                         for j in range(self.N_photens)])

        ## Initial populations
        self.rho_core_0 = self.M_CE           # Initially fully occupied
        self.rho_free_0 = 0              # Initially not occupied
        self.rho_VB_0 = self.get_initial_valence_occupation() * self.M_VB       # Initially occupied up to Fermi Energy
        self.enpool_T_0 = self.kB*temperature # Initial thermal energy of electron system
        self.enpool_free_0 = 0           # Initial energy of kinetic electrons, Initally zero
        self.rho_Ej_0 = [0]*N_photens    # Deviations from thermal distributions are initially zero in equilibrium
        
        # Calculate initial contribution of the thermal distribution at the resonant photon energies
        self.fermi_el_at_T0_Ej = sim.fermi(self.enpool_T_0, self.rho_VB_0,self.E_j, self.E_f)
        
               
        # This vector contains all parameters that are tracked over time
        self.state_vector_0 = np.array([
            self.rho_core_0,
            self.rho_free_0,
            self.rho_VB_0,
            self.enpool_T_0,
            self.enpool_free_0,
            *self.rho_Ej_0] * self.Nsteps_z).reshape(self.Nsteps_z,self.states_per_voxel)
        
        print('Initiated simulation parameters.\n Number of tracked parameters: ', np.size(self.state_vector_0))
        
    def get_resonant_states(self, E, resonant_width, npoints = 10):
        """
        Returns the number of states resonant to the photon energy E, 
        assuming a certain resonant width.
        The DoS data stored in the file should be normalized to unity.
        A re-sampling is done to prevent any funny businness due to the sampling density of the DoS.
        """
        Emin = E-resonant_width
        Emax = E+resonant_width

        X = np.linspace(Emin, Emax, npoints)
        Y = np.interp(X,self.DoSdata['x'],self.DoSdata['y'])
        return np.trapz(y = Y, x= X)
    
    def get_initial_valence_occupation(self):
        """
        This one just calculates the initial valence state population
        """
        x = self.DoSdata['x']
        y = self.DoSdata['y']
        return np.trapz(y = y[x<0], x= x[x<0])

    def pulse_profiles(self,t):
        """
        For now this returns a simple Gaussian profile for each photon energy Ej.
        A call costs 9.7 µs on jupyterhub for two Energies at one time - perhaps this can be 
        reduced by using interpolation between a vector in the workspace.
        """
        return self.I0 * np.exp(-0.5*((t-self.t0)/self.tdur_sig)**2) * 1/(np.sqrt(2*np.pi)*self.tdur_sig)
    
    def get_fermi(occup):
        """
        Calculates the current Fermi energy change from the current occupation if the valence system
        """
        if (occup > 1) or (occup < 0):
            raise RuntimeError('Occupation outside possible bounds!')
        dx = self.DoSdata['x'][1]-self.DoSdata['x'][0]
        return np.interp(occup, self.DoSdata['osi'], self.DoSdata['x']-dx)

        
## Main Simulation

class XNLsim:
    def __init__(self, par):
        self.par = par
        self.par.make_derived_params(self)

        # initiate storing intermediate results
        self.call_counter = 0
        self.all_N_Ej_z = []
        self.thermal_occupations = []

    """
    Processes
    """

    # f(T,i)
    def fermi(self, T, rho_VB, E_j, E_f, j = np.s_[:]):
        return (rho_VB/self.par.M_VB) * (self.par.M_Ej[j] /(np.exp((E_j-E_f)/T)+1))
    def calc_thermal_occupations(self, state_vector):
        """
        It appears favorable to do this separately at the beginning of the call
        :param state_vector:
        :return thermal_occupations:  For this call
        """
        thermal_occupations = np.empty((self.par.Nsteps_z,self.par.N_photens))

        # Loop through sample depth
        for iz in range(self.par.Nsteps_z):
            rho_VB, T = state_vector[iz,2:4]
            thermal_occupations[iz,:] = self.fermi(T, rho_VB, self.par.E_j,  self.par.E_f)
        return thermal_occupations

    # Resonant interaction
    def proc_res_inter_Ej(self, N_Ej, rho_CE,  rho_Ej):
        return ( (rho_CE/self.par.M_CE) - (rho_Ej+ self.thermal_occupations[self.call_counter] -\
                                            self.par.fermi_el_at_T0_Ej) / self.par.M_Ej)\
                                            *(N_Ej/self.par.lambda_res_Ej)

    # Nonresonant interaction
    def proc_nonres_inter(self, N_Ej, rho_VB, rho_Ej):
        return (((rho_VB - self.par.rho_VB_0) + np.sum(rho_Ej))*N_Ej)/(self.par.M_VB * self.par.lambda_nonres)

    # Core-hole decay
    def proc_ch_decay(self, rho_CE, rho_VB, rho_Ej):
        return (self.par.M_CE-rho_CE)*((rho_VB-self.par.rho_VB_0)+np.sum(rho_Ej))/(self.par.M_VB*self.par.tau_CH)

    # Electron Thermalization
    def proc_el_therm(self, rho_Ej):
        return rho_Ej/self.par.tau_therm

    # Free electron scattering
    def proc_free_scatt(self, rho_free, rho_VB, rho_Ej):
        return ((rho_VB+np.sum(rho_Ej))*rho_free)/(self.par.rho_VB_0 * self.par.tau_free)

    # Mean energy of kinetic electrons
    def mean_free_el_energy(self, rho_free, E_free):
        if rho_free == 0:
            return 0
        return E_free/rho_free

    # Mean energy of electrons in the valence system
    def mean_valence_energy(self, rho_VB, rho_Ej, E_f):
        return (rho_VB * E_f + np.sum(rho_Ej*self.par.E_j))/(rho_VB + np.sum(rho_Ej))

    # unpacks state vector and calls all the process functions
    def calc_processes(self, N_Ej, states):
        """
        Calculates all the processes for a given z
        """
        rho_CE, rho_free, rho_VB, T, E_free = states[0:5]
        rho_Ej       = states[5:]
        res_inter    = self.proc_res_inter_Ej(N_Ej, rho_CE, rho_Ej)
        nonres_inter = self.proc_nonres_inter(N_Ej, rho_VB, rho_Ej)
        ch_decay     = self.proc_ch_decay(rho_CE, rho_VB, rho_Ej)
        el_therm     = self.proc_el_therm(rho_Ej)
        el_scatt     = self.proc_free_scatt(rho_free, rho_VB, rho_Ej)
        mean_free    = self.mean_free_el_energy(rho_free, E_free)
        mean_valence = self.mean_valence_energy(rho_VB, rho_Ej, self.par.E_f)
        return res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence

    def rate_N_dz_j_direct(self, N_Ej, states):
        """
        Calculates only dN/dz for a given z.
        This one is for the directly coded light propagation
        """
        rho_CE, rho_free, rho_VB, T, E_free = states[0:5]
        rho_Ej = states[5:]
        res_inter = self.proc_res_inter_Ej(N_Ej, rho_CE,  rho_Ej)
        return -res_inter


    """
    Rates - time derivatives 
    """

    def rate_CE(self, res_inter, ch_decay):
        return ch_decay - np.sum(res_inter)

    def rate_free(self, nonres_inter, ch_decay, el_scatt):
        return np.sum(nonres_inter) + ch_decay - el_scatt

    def rate_VB(self, res_inter, nonres_inter, el_therm, ch_decay, el_scatt):
        return np.sum(-nonres_inter + el_therm) -2*ch_decay + el_scatt

    def rate_E_j(self, res_inter, el_therm):
        return res_inter - el_therm

    def rate_E_free(self, nonres_inter, ch_decay, el_scatt, mean_free, mean_valence):
        return np.sum(nonres_inter*(self.par.E_j-mean_valence))+(ch_decay*mean_valence) - (el_scatt*mean_free)

    def rate_T(self, el_therm, el_scatt, mean_free, mean_valence):
        return np.sum(el_therm*(self.par.E_j-mean_valence)) + el_scatt*mean_free


    """
    Propagation of photons through the sample
    """
    def z_dependence(self, t, state_vector):
        def zstep_euler(self, N, state_vector, iz):
            return N + self.rate_N_dz_j_direct(N, state_vector[iz, :]) * self.par.zstepsize
        def double_zstep_RK(self, N, state_vector, iz):
            """
            Since I only know the states at specific points in z, I cheat by doubling the effective z step.
            """
            k1 = self.rate_N_dz_j_direct(N, state_vector[iz, :])
            k2 = self.rate_N_dz_j_direct(N + self.par.zstepsize * k1, state_vector[iz + 1, :])
            k3 = self.rate_N_dz_j_direct(N + self.par.zstepsize * k2, state_vector[iz + 1, :])
            k4 = self.rate_N_dz_j_direct(N + self.par.zstepsize * 2 * k3, state_vector[iz + 2, :])
            return N + 0.3333333333333333 * self.par.zstepsize * (k1 + 2 * k2 + 2 * k3 + k4)

        # get current photon irradiation:
        N_Ej_z = np.zeros((self.par.Nsteps_z,self.par.N_photens))

        N_Ej_z[0,:] = self.par.pulse_profiles(t) # momentary photon densities at time t for each photon energy onto the sample

        js = np.arange(self.par.N_photens)
        # Z-loop for every photon energy
        N_Ej_z[1,:] = zstep_euler(self, N_Ej_z[0,js], state_vector, 0) # First step with euler
        for iz in range(2,self.par.Nsteps_z,1):
            N_Ej_z[iz,:] = double_zstep_RK(self, N_Ej_z[iz-2,js], state_vector, iz-2)

        return N_Ej_z


    """
    Main differential describing the time evolution of voxels
    """
    def time_derivative(self, t, state_vector_flat):
        # Reshape the state vector into sensible dimension
        state_vector = state_vector_flat.reshape(self.par.Nsteps_z,self.par.states_per_voxel)

        self.thermal_occupations.append(self.calc_thermal_occupations(state_vector))

        # Initiate empty variables
        derivatives = np.empty((self.par.Nsteps_z,self.par.states_per_voxel))

        # Calculate photon transmission as save it
        N_Ej_z = self.z_dependence(t, state_vector)
        self.all_N_Ej_z.append(N_Ej_z)

        # Loop through sample depth
        for iz in range(self.par.Nsteps_z):
            res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence = \
                self.calc_processes(N_Ej_z[iz,:], state_vector[iz,:])

            derivatives[iz,:] = np.array([
            self.rate_CE(res_inter, ch_decay),
            self.rate_free(nonres_inter, ch_decay, el_scatt),
            self.rate_VB(res_inter, nonres_inter, el_therm, ch_decay, el_scatt),
            self.rate_T(el_therm, el_scatt, mean_free, mean_valence),
            self.rate_E_free(nonres_inter, ch_decay, el_scatt, mean_free, mean_valence),
            *self.rate_E_j(res_inter, el_therm) ])

        self.call_counter += 1
        return derivatives.flatten()