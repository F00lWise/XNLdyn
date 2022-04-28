import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import warnings

# Import all the parameters defined in the params file and processed in process_params
from .params import *


def check_bounds(value, min=0, max=1., message=''):
    if np.any(value < min):
        string = f'Found value up to {np.min(value[value < min]) - min:.3e} under minimum of {min}.' + message
        warnings.warn(string)
    if np.any(value > max):
        string = f'Found values {np.max(value[value > max]) - max:.3e} over maximum of {max}.' + message
        warnings.warn(string)


class XNLpars:
    def __init__(self):
        ## Some constants
        self.kB = 8.617333262145e-5  # Boltzmann Konstant / eV/K
        self.lightspeed = 299792458  # m/s
        self.hbar = 6.582119569e-15  # eV s
        self.echarge = 1.602176634e-19  # J/eV

        ## Here I just "package" the variables from the params file into class attributes
        self.Nsteps_z = Nsteps_z  # Steps in Z
        self.N_photens = N_photens  # Number of distict resonant photon energies

        self.N_j = N_j

        ## Sample thickness
        self.Z = Z  # nm

        self.atomic_density = atomic_density  # atoms per nm³
        self.photon_bandwidth = photon_bandwidth  # The energy span of the valence band that each resonant energy interacts with./ eV
        self.temperature = temperature  # Kelvin

        ## Electronic state numbers per atom
        self.core_states = core_states
        self.total_valence_states = total_valence_states
        self.DoS_shapefile = DoS_shapefile

        ## Rates and cross sections
        self.tau_CH = tau_CH
        self.tau_th = tau_th
        self.lambda_res_Ei = np.array(lambda_res_Ei)
        self.lambda_nonres = lambda_nonres

        ## Fermi Energy
        self.E_f = E_f

        ## Incident photon profile - here with the dimension i in range(N_photens)
        self.I0_i = np.array(I0)
        self.t0_i = np.array(t0)
        self.tdur_sig_i = np.array(tdur_sig)
        self.E_i = np.array(E_i)

        assert (N_photens == len(I0) == len(t0) == len(tdur_sig) == len(E_i) == len(lambda_res_Ei)), \
            'Make sure all photon pulses get all parameters!'

    def make_derived_params(self, sim):
        ## now some derived quantities
        self.zstepsize = self.Z / self.Nsteps_z
        self.zaxis = np.arange(0, self.Z, self.zstepsize)

        self.E_i = np.array(self.E_i) - self.E_f  # self.E_i becomes relative to Fermi edge

        # Energy Axis
        self.E_j, self.enax_j_edges = self.make_valence_energy_axis(self.N_j)
        # Indizes where E_j == E_i
        self.resonant = np.array([(True if E_j in self.E_i else False) for E_j in self.E_j])

        ## Expanding the incident photon energies so that they match the tracked energies
        self.lambda_res_Ej = self.I0 = self.t0 = self.tdur_sig = np.zeros(self.N_j, dtype = np.float64)
        self.lambda_res_Ej[self.resonant] = self.lambda_res_Ei
        self.I0[self.resonant] = self.I0_i
        self.t0[self.resonant] = self.t0_i
        self.tdur_sig[self.resonant] = self.tdur_sig_i

        ## Load DoS data
        ld = np.load(self.DoS_shapefile)
        self.DoSdata = {}
        self.DoSdata['x'] = ld[:, 0]
        self.DoSdata['y_atomar'] = ld[:, 1]
        self.DoSdata['y'] = self.DoSdata['y_atomar']  # * self.M_VB
        self.DoSdata['osi'] = np.array([np.trapz(self.DoSdata['y'][:i], x=self.DoSdata['x'][:i]) for i in range(
            len(self.DoSdata['x']))])  # one-sided integral for calculating Fermi energy

        ## Multiplicities - these are global for all t and z
        self.M_core = self.atomic_density * self.core_states
        self.M_VB = self.atomic_density * self.total_valence_states
        self.m_j = self.get_resonant_states(self.M_VB)

        ## Initial populations
        self.R_core_0 = self.M_core  # Initially fully occupied
        self.R_free_0 = 0  # Initially not occupied
        self.E_free_0 = 0  # Initial energy of kinetic electrons, Initally zero
        self.rho_j_0 = self.m_j * sim.fermi(self.kB * temperature)  # occupied acording to initial temperature

        ## derived from these
        self.R_VB_0 = np.sum(self.rho_j_0)  # Initially occupied up to Fermi Energy
        self.enpool_T_0 = self.kB * temperature  # Initial thermal energy of the average valence electron

        # This vector contains all parameters that are tracked over time
        self.states_per_voxel = 3 + self.N_j  # Just the number of entries for later convenience
        self.state_vector_0 = np.array([
                                   self.R_core_0,
                                   self.R_free_0,
                                   self.E_free_0,
                                   *self.rho_j_0] * self.N_j).reshape(self.N_j, self.states_per_voxel)

    def get_resonant_states(self, M_VB):
        """
        Returns the partial state densities at energies Ej so that sum(m_j) = M_VB
        """
        m_j = np.empty(self.N_j)
        for j in range(self.E_j):
            Emin = self.enax_j_edges[j]
            Emax = self.enax_j_edges[j+1]

            X = np.linspace(Emin, Emax, 10)
            Y = np.interp(X, self.DoSdata['x'], self.DoSdata['y'])

            m_j[j] = np.trapz(y=Y, x=X)
        m_j = m_j / np.sum(m_j) # make sure its normalized to the sum
        m_j = m_j * M_VB
        return m_j

    # def get_initial_valence_occupation(self):
    #     """
    #     This one just calculates the initial valence state population
    #     """
    #     x = self.DoSdata['x']
    #     y = self.DoSdata['y']
    #     return np.trapz(y=y[x < 0], x=x[x < 0])

    def pulse_profiles(self, t):
        """
        For now this returns a simple Gaussian profile for each photon energy Ej.
        A call costs 9.7 µs on jupyterhub for two Energies at one time - perhaps this can be
        reduced by using interpolation between a vector in the workspace.
        """
        return self.I0 * np.exp(-0.5 * ((t - self.t0) / self.tdur_sig) ** 2) * 1 / (np.sqrt(2 * np.pi) * self.tdur_sig)

    def make_valence_energy_axis(self, N_j: np.int, min=-6, finemax=4, max=20):
        """
        Creates an energy axis for the valence band, namely
            self.E_j
        and its edgepoints
            self.enax_j_edges
        Energies are relative to the fermi-level. 3/4 of all points fall into the range (min, finemax)
        Makes sure that the energies E_i correspond to a point in E_j and
        drops the closest points to keep the number N_j.
        :param N_j:
        :param min:
        :param finemax:
        :param max:
        :return:
        """
        N_j_fine = int(N_j * 3 / 4)
        N_j_coarse = int(N_j - N_j_fine)

        # Midpoints!
        enax_j_fine = list(np.linspace(min, finemax, N_j_fine))
        dE_fine = enax_j_fine[1] - enax_j_fine[0]
        enax_j_coarse = list(np.linspace(finemax + dE_fine, max, N_j_coarse))
        # make sure that resonant energies are in there
        enax_j = np.concatenate((enax_j_fine, enax_j_coarse))
        good_js = list(np.ones(enax_j.shape, dtype=bool))
        for i in range(self.N_photens):
            deltas = enax_j - self.E_i[i]
            good_js[np.argmin(deltas)] = False  # drop the one closest to E_i
        enax_j = np.sort(np.concatenate((enax_j[good_js], self.E_i)))

        if not len(enax_j) == N_j:
            warnings.warn(
                'Energy Axis turned out longer or shorter than planne. Are resonant energies very close together?')
            self.N_j = len(enax_j)
        def edgepoints(middles):
            """ Opposite of midpoints """
            edges = np.empty(middles.shape[0] + 1)
            edges[1:-1] = (middles[1:] + middles[:-1]) / 2
            edges[0] = middles[0] - (middles[1] - middles[0]) / 2
            edges[-1] = middles[-1] + (middles[-1] - middles[-2]) / 2
            return edges

        return enax_j, edgepoints(enax_j)


## Main Simulation

class XNLsim:
    def __init__(self, par, DEBUG=False, atol=1e-18):
        self.DEBUG = DEBUG
        self.intermediate_plots = False
        self.par = par
        self.par.make_derived_params(self)

        # initiate storing intermediate results
        self.call_counter = 0
        self.thermal_occupations = None

        # Tolerance - variables are rounded to this value if they become very small to avoid float underfolows
        # self.atol = atol # I think this might not be needed

    """
    Processes
    """

    def run(self, t_span, method, rtol, atol, plot=False, return_full_solution=False, set_debug=False):
        """
        Run the simulation once for a z-stack

        :param t_span: Time axis to simulate within, limits in femtoseconds
        :param method: Use 'RK45' for or 'DOP853' (tha latter for very small error goals)
        :param rtol: Relative error goal
        :param atol: Absolute error goal
        :param plot: Boolean to plot results
        :return: incident_pulse_energies, transmitted_pulse_energies
        """
        ### Solve Main problem
        sol = solve_ivp(self.time_derivative, t_span=t_span, \
                        dense_output=True, y0=self.par.state_vector_0.flatten(), method=method, rtol=rtol,
                        atol=atol)  # DOP853 or RK45

        soly = sol.y.reshape((self.par.Nsteps_z, self.par.states_per_voxel, len(sol.t)))

        ### Since they weren't saved, calculate transmission again
        sol_photon_densities = np.zeros((self.par.Nsteps_z, self.par.N_photens, len(sol.t)))
        for it, t in enumerate(sol.t):
            sol_photon_densities[:, :, it] = self.z_dependence(t, soly[:, :, it])

        incident_pulse_energies = np.trapz(sol_photon_densities[0, :, :], x=sol.t)
        transmitted_pulse_energies = np.trapz(sol_photon_densities[-1, :, :], x=sol.t)

        if plot:
            self.plot_results(sol, sol_photon_densities)

        if return_full_solution:
            return incident_pulse_energies, transmitted_pulse_energies, sol
        else:
            return incident_pulse_energies, transmitted_pulse_energies

    # Calcf(T,i)
    def fermi(self, T: float, E_j = None):
        """
        Returns the fermi distribution (between 0 and 1) for the energies E_j, which default to self.E_j
        :param T:
        :param E_j:
        :return: relative_occupations
        """
        if E_j is None:
            E_j = self.E_j

        # Due to the exponential I get a floating point underflow when calculating the fermi distribution naively, hence the extra effort
        fermi_distr = np.zeros(E_j.shape)
        energy_ratios = E_j / T
        calculatable = np.abs(energy_ratios) < 15
        fermi_distr[calculatable] = 1 / (np.exp(energy_ratios[calculatable]) + 1)
        fermi_distr[energy_ratios < -15] = 1
        fermi_distr[energy_ratios > 15] = 0
        if self.DEBUG:
            check_bounds(fermi_distr, message='Fermi distribution in fermi()')
        return fermi_distr

    # def calc_thermal_occupations(self, state_vector):
    #     """
    #     It appears favorable to do this separately at the beginning of the call
    #     :param state_vector:
    #     :return thermal_occupations:  For this call
    #     """
    #     thermal_occupations = np.zeros((self.par.Nsteps_z, self.par.N_photens))
    #
    #     # Loop through sample depth
    #     for iz in range(self.par.Nsteps_z):
    #         rho_VB = state_vector[iz, 2]
    #         T = state_vector[iz, 3] / self.par.M_VB  # need the thermal energy per electron
    #         thermal_occupations[iz, :] = self.fermi(T, rho_VB, self.par.E_j, self.par.E_f)
    #
    #     if self.intermediate_plots:
    #         self.plot_thermal_occupations(thermal_occupations)
    #         plt.show(block=False)
    #         plt.pause(0.1)
    #
    #     if self.DEBUG:
    #         for j in range(self.par.N_photens):
    #             check_bounds(thermal_occupations[:, j], 0, self.par.E_j[j],
    #                          message='Just computed an unrealistic thermal occupation')
    #     return thermal_occupations

    # Resonant interaction
    def proc_res_inter_Ej(self, N_Ej, R_core, rho_j):
        core_occupation = (R_core / self.par.M_core)
        valence_occupation = rho_j/self.par.m_j # relative to the states at that energy
        if self.DEBUG:
            check_bounds(core_occupation, message='Valence occupation in proc_res_inter_Ej()')
            check_bounds(valence_occupation, message='Valence occupation in proc_res_inter_Ej()')
        return np.outer((core_occupation - valence_occupation), (N_Ej / self.par.lambda_res_Ej)) # returns j,i

    # Nonresonant interaction
    def proc_nonres_inter(self, N_Ej, rho_j):
        valence_occupation = rho_j/self.par.R_VB_0 # relative to the number valence states in the ground state
        if self.DEBUG: check_bounds(valence_occupation, 0, 1, message='valence occupation deviation in proc_nonres_inter()')
        return (valence_occupation.T* N_Ej.T).T / self.par.lambda_nonres # returns j,i

    # Core-hole decay
    def proc_ch_decay(self, R_core, rho_j):
        core_holes = (self.par.M_core - R_core)/ self.par.M_core
        if self.DEBUG:
            check_bounds(core_holes, message='Core holes in proc_ch_decay()')
        return core_holes * rho_j / self.par.tau_CH

    # Electron Thermalization
    def proc_el_therm(self, rho_j, r_j):
        return (r_j - rho_j) / self.par.tau_therm

    # Free electron scattering
    def proc_free_scatt(self, R_free):
        return R_free / self.par.tau_free

    # Mean energy of kinetic electrons
    def mean_free_el_energy(self, R_free, E_free):
        empty = (R_free < 1e-6) # hardcoded precision limit of 1 µeV
        mean_free = np.empty(R_free.shape)
        mean_free[~empty] = E_free[~empty] / R_free[~empty]
        mean_free[empty] = 0
        return mean_free

    # Mean energy of electrons in the valence system
    def mean_valence_energy(self, rho_j, E_free, R_free):
        """
        I am not sure yet which electron's energy contributes. My instinct says only valence,
        Martin says also free AND core. Here is valence and Free. Check!
        :param rho_j:
        :param E_free:
        :param R_free:
        :return:
        """
        total_energy = np.sum(self.par.E_j * rho_j) + E_free
        return total_energy / (np.sum(rho_j, axis=1) + R_free)


    # unpacks state vector and calls all the process functions
    def calc_processes(self, N_Ej, states):
        """
        Calculates all the processes for all z
        dimeninality of each j-resolved variable is [iz,j]
        """

        R_core = states[:, 0]
        R_free = states[:, 1]
        E_free = states[:, 2]
        rho_j = states[:, 3:]

        T = self.mean_valence_energy(rho_j, E_free, R_free)
        r_j = self.fermi(T) * self.par.m_j

        res_inter = self.proc_res_inter_Ej(N_Ej, R_core, rho_j)
        nonres_inter = self.proc_nonres_inter(N_Ej, rho_j)
        ch_decay = self.proc_ch_decay(R_core, rho_j)
        el_therm = self.proc_el_therm(rho_j, r_j)
        el_scatt = self.proc_free_scatt(R_free)
        mean_free = self.mean_free_el_energy(R_free, E_free)
        mean_valence = self.mean_valence_energy(rho_j, E_free, R_free)
        return res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence

    def rate_N_dz_j_direct(self, N_Ej, states):
        """
        Calculates only dN/dz for a given z.
        This one is for the directly coded light propagation
        """
        R_core, R_free, E_free = states[0:3]
        rho_j = states[3:]
        core_occupation = (R_core / self.par.M_core)
        valence_occupation = rho_j/self.par.m_j # relative to the states at that energy
        return - (core_occupation.T - valence_occupation.T).T * (N_Ej / self.par.lambda_res_Ej)

    """
    Rates - time derivatives 
    """
    def rate_j(self, res_inter, nonres_inter, el_therm, ch_decay, rho_j, R_VB):
        return res_inter - np.sum(nonres_inter, axis = 2) - ch_decay - (rho_j * np.sum(ch_decay, axis = 1)/R_VB) + el_therm

    def rate_core(self, res_inter, ch_decay):
        return np.sum(ch_decay, axis=1) - np.sum(res_inter, axis=1)

    def rate_free(self, nonres_inter, ch_decay, el_scatt):
        return np.sum(nonres_inter, axis=1) + np.sum(ch_decay, axis = 1) - el_scatt

    # def rate_E_free(self, R_free):
        # , nonres_inter, ch_decay, el_scatt, mean_free, mean_valence):
        # energies_unfolded = np.outer(np.ones(self.par.Nsteps_z), self.par.E_j).T
        # return np.sum(nonres_inter*(self.par.E_j-), axis = (1,2))
        #     np.sum(nonres_inter * (energies_unfolded - mean_valence).T, axis=1) + (ch_decay * mean_valence) - (
        #         el_scatt * mean_free)


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
        N_Ej_z = np.zeros((self.par.Nsteps_z, self.par.N_j))

        N_Ej_z[0, :] = self.par.pulse_profiles(t)  # momentary photon densities at time t for each photon energy

        js = np.arange(self.par.N_photens)
        # Z-loop for every photon energy
        N_Ej_z[1, :] = zstep_euler(self, N_Ej_z[0, js], state_vector, 0)  # First step with euler
        for iz in range(2, self.par.Nsteps_z):
            N_Ej_z[iz, :] = double_zstep_RK(self, N_Ej_z[iz - 2, js], state_vector, iz - 2)

        return N_Ej_z

    """
    Main differential describing the time evolution of voxels
    """

    def time_derivative(self, t, state_vector_flat):
        # Reshape the state vector into sensible dimension
        state_vector = state_vector_flat.reshape(self.par.Nsteps_z, self.par.states_per_voxel)
        check_bounds(state_vector[:, 3], 0, np.inf,
                     message='Temperature in time_derivative.')  # The temperature must never become negative
        # state_vector[state_vector<self.atol] = 0
        # state_vector[:, 3][state_vector[:, 3]<0] = 0 # Dirty fix since this seems to happen! Look into!

        self.thermal_occupations = self.calc_thermal_occupations(state_vector)

        # Calculate photon transmission as save it
        N_Ej_z = self.z_dependence(t, state_vector)

        # No longer loop through sample depth
        res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence = \
            self.calc_processes(N_Ej_z[:, :], state_vector[:, :])

        derivatives = np.empty(state_vector.shape)
        derivatives[:, 0] = self.rate_CE(res_inter, ch_decay)
        derivatives[:, 1] = self.rate_free(nonres_inter, ch_decay, el_scatt)
        derivatives[:, 2] = self.rate_VB(res_inter, nonres_inter, el_therm, ch_decay, el_scatt)
        derivatives[:, 3] = self.rate_T(el_therm, el_scatt, mean_free, mean_valence)
        derivatives[:, 4] = self.rate_E_free(nonres_inter, ch_decay, el_scatt, mean_free, mean_valence)
        derivatives[:, 5:] = self.rate_E_j(res_inter, el_therm)

        # Debug plotting
        if self.intermediate_plots:
            if np.mod(self.call_counter, 20) == 0:
                self.plot_z_dependence(N_Ej_z)
                self.plot_occupancies(state_vector)
                self.plot_derivatives(derivatives)
                plt.show(block=False)
                plt.pause(0.1)

        self.call_counter += 1
        return derivatives.flatten()

    def plot_z_dependence(self, N_Ej_z):
        if not 'figure_z' in dir(self):
            self.figure_z = plt.figure()
            self.axis_z = plt.gca()
        else:
            plt.sca(self.axis_z)

        plt.plot(self.par.zaxis, N_Ej_z)
        plt.xlabel('z')
        plt.ylabel('Photon density')
        self.axis_z.set_title(f'Z-Dependence')

    def plot_occupancies(self, state_vector):
        if not 'figure_occ' in dir(self):
            self.figure_occ = plt.figure()
            self.axis_occ = plt.gca()

        else:
            plt.sca(self.axis_occ)
            self.axis_occ.clear()
        plt.plot(self.par.zaxis, (state_vector[:, 0] / self.par.M_CE) - 1, label='Core occupation variation')
        plt.plot(self.par.zaxis, state_vector[:, 1], label='Free electrons')
        plt.plot(self.par.zaxis, (state_vector[:, 2] - self.par.rho_VB_0) / self.par.M_VB,
                 label='VB occupation variation')
        plt.plot(self.par.zaxis, state_vector[:, 3] - (300 * self.par.kB), label='T')
        plt.plot(self.par.zaxis, state_vector[:, 4], label='E')
        plt.plot(self.par.zaxis, state_vector[:, 5:] / self.par.M_Ej, label='Resonant occupations')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('Photon density')
        self.axis_occ.set_title('Occupancies')

    def plot_derivatives(self, derivatives):
        if not 'figure_der' in dir(self):
            self.figure_der = plt.figure()
            self.axis_der = plt.gca()
        else:
            plt.sca(self.axis_der)
            self.axis_der.clear()
        der = derivatives.reshape(self.par.Nsteps_z, self.par.states_per_voxel)
        plt.plot(der[:, 0] / self.par.M_CE, label='Core derivative')
        plt.plot(der[:, 1], label='Free electrons')
        plt.plot(der[:, 2] / self.par.M_VB, label='VB derivative')
        plt.plot(der[:, 3], label='T')
        plt.plot(der[:, 4], label='E')
        plt.plot(der[:, 5:] / self.par.M_Ej, label='Resonant derivative')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('Photon density')
        self.axis_der.set_title('Derivatives')

    def plot_thermal_occupations(self, thermal_occupations=None):
        if not 'figure_therm' in dir(self):
            self.figure_therm = plt.figure()
            self.axis_therm = plt.gca()
        else:
            plt.sca(self.axis_therm)
            self.axis_therm.clear()

        if thermal_occupations is None:
            thermal_occupations = self.thermal_occupations

        self.axis_therm.set_title('Thermal Occupations')
        plt.plot(thermal_occupations)

    def plot_results(self, sol, sol_photon_densities):
        PAR = self.par
        ## Plotting
        soly = sol.y.reshape((PAR.Nsteps_z, PAR.states_per_voxel, len(sol.t)))
        sol_core = soly[:, 0, :]
        sol_free = soly[:, 1, :]
        sol_VB = soly[:, 2, :]
        sol_T = soly[:, 3, :]
        sol_Efree = soly[:, 4, :]
        sol_Ej = soly[:, 5:, :]

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        plt.sca(axes[0, 0])
        plt.title('State occupation changes')
        plt.plot(sol.t, 1 - (np.mean(sol_core, 0) / PAR.M_CE), label='Core holes')
        plt.plot(sol.t, 1 - (sol_core[0] / PAR.M_CE), label='Core holes [0]')

        plt.plot(sol.t, (np.mean(sol_VB, 0) - PAR.rho_VB_0) / PAR.M_VB, label='Valence band occupation')
        plt.plot(sol.t, (sol_VB[0] - PAR.rho_VB_0) / PAR.M_VB, label='Valence @surface')

        plt.plot(sol.t, np.mean(sol_Ej, 0).T / PAR.M_Ej,
                 label=[f'{PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])

        plt.ylabel('Occupation')
        plt.xlabel('t (fs)')
        plt.legend()

        plt.sca(axes[0, 1])
        plt.title('Kinetic electrons')

        plt.plot(sol.t, np.mean(sol_free, 0), label='Kinetic electrons')
        plt.plot(sol.t, sol_free[0], label='Surface')

        plt.ylabel('Occupation')
        plt.xlabel('t (fs)')
        plt.legend()

        plt.sca(axes[1, 0])
        plt.title('Energies Averaged over z')
        plt.plot(sol.t, np.mean(sol_T, 0), label='Valence Thermal Energy')
        plt.plot(sol.t, np.mean(sol_Efree, 0), label='Free Electron Energy')
        plt.plot(sol.t, sol_T[0], label='Valence @ Surface')
        plt.plot(sol.t, sol_Efree[0], label='Free surface')
        plt.ylabel('E (eV)')
        plt.xlabel('t (fs)')
        plt.legend()

        plt.sca(axes[1, 1])
        plt.title('Photons')
        plt.xlabel('t (fs)')

        plt.plot(sol.t, sol_photon_densities[-1, :, :].T,
                 label=[f'Transmitted {PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in
                        range(PAR.N_photens)])

        plt.plot(sol.t, sol_photon_densities[0, :, :].T,
                 label=[f'Incident {PAR.E_j[i]:.0f} eV,  {PAR.lambda_res_Ej[i]:.0f} nm' for i in range(PAR.N_photens)])
        plt.legend()
        plt.ylabel('T')
        plt.xlabel('t (fs)')

        plt.tight_layout()
        plt.show()
