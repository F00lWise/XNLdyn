import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sc
from scipy.integrate import solve_ivp
import numpy as np
import warnings
import lmfit

# Import all the parameters defined in the params file and processed in process_params
from .params import *

RTOL = 1e-5
def check_bounds(value, min=0 - RTOL, max=1. + RTOL, message=''):
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
        self.M_core = core_states
        #self.M_VB = total_valence_states # define as sum(m_j)
        self.R_VB_0 = valence_GS_occupation
        self.DoS_shapefile = DoS_shapefile

        ## Rates and cross sections
        self.tau_CH = tau_CH
        self.tau_th = tau_th
        self.tau_free = tau_free
        self.lambda_res_Ei = np.array(lambda_res_Ei)
        self.lambda_nonres = lambda_nonres

        ## Fermi Energy
        self.E_f = E_f

        ## Incident photon profile - here with the dimension i in range(N_photens)
        self.I0_i = np.array(I0)
        self.t0_i = np.array(t0)
        self.tdur_sig_i = np.array(tdur_sig)
        self.E_i_abs = np.array(E_i)
        self.E_i = np.empty(N_photens) # defined below

        assert (N_photens == len(I0) == len(t0) == len(tdur_sig) == len(E_i) == len(lambda_res_Ei)), \
            'Make sure all photon pulses get all parameters!'

    def make_derived_params(self, sim):
        ## now some derived quantities
        self.zstepsize = self.Z / self.Nsteps_z
        self.zaxis = np.arange(0, self.Z, self.zstepsize)

        self.E_i = np.array(self.E_i_abs) - self.E_f  # self.E_i becomes relative to Fermi edge
        # Energy Axis
        self.E_j, self.enax_j_edges = self.make_valence_energy_axis(self.N_j)
        # Indizes where E_j == E_i
        self.resonant = np.array([(True if E_j in self.E_i else False) for E_j in self.E_j])


        ## Expanding the incident photon energies so that they match the tracked energies
        self.lambda_res_Ej = np.zeros(self.N_j, dtype = np.float64)
        self.I0 = np.zeros(self.N_j, dtype = np.float64)
        self.t0 = np.zeros(self.N_j, dtype = np.float64)
        self.tdur_sig = np.zeros(self.N_j, dtype = np.float64)
        self.lambda_res_Ej[self.resonant] = self.lambda_res_Ei
        self.lambda_res_Ej_inverse = np.zeros(self.lambda_res_Ej.shape, dtype=np.float64)
        self.lambda_res_Ej_inverse[self.resonant] = 1/self.lambda_res_Ej[self.resonant]
        
        self.I0[self.resonant] = self.I0_i       
        self.t0[self.resonant] = self.t0_i
        self.tdur_sig[self.resonant] = self.tdur_sig_i
        
        ## Load DoS data
        ld = np.load(self.DoS_shapefile)
        DoSdata = {}
        DoSdata['enax'] = ld[:, 0]
        DoSdata['DoS'] = ld[:, 1]
        DoS_raw = np.interp(self.E_j, DoSdata['enax'][DoSdata['enax'] < 3], DoSdata['DoS'][DoSdata['enax'] < 3]) # inteprolate to general enaxis, but extend everything beyond +3eV to constant

        ## mj are normalized by the ground state population
        # Best way to do it: self.DoS = self.R_VB_0* DoS_raw / np.trapz(np.append(DoS_raw[self.E_j<=0],np.interp(0,self.E_j,DoS_raw)), np.append(self.E_j[self.E_j<=0],0))
        # Faster and consistent way to do it:
        #self.DoS = DoS_raw * self.R_VB_0 / np.trapz(DoS_raw[self.E_j<=0], self.E_j[self.E_j<=0])

        #        self.m_j = np.array([self.get_resonant_states(self.enax_j_edges[i], self.enax_j_edges[i+1]) for i, _ in enumerate(self.E_j)])
        enax_dE = self.enax_j_edges[1:] - self.enax_j_edges[:-1]
        #DoS = DoS / np.sum(DoS)
        self.m_j = DoS_raw * enax_dE  # scale with energy step size
        occupied = (np.sum(self.m_j[self.E_j<0])+ 0.5*self.m_j[self.E_j==0])/np.sum(self.m_j) # part that is occupied in GS
        self.m_j = self.m_j / np.sum(self.m_j) # normalize to one to be sure
        self.m_j*= valence_GS_occupation / occupied # scale to ground state occupation
        self.M_VB = np.sum(self.m_j)
        #self.m_j = m_j * 10 / np.sum(m_j[enax < 0])

        self.FermiSolver = FermiSolver(self, self.m_j)

        ## Initial populations
        self.R_core_0 = self.M_core  # Initially fully occupied
        self.R_free_0 = 0  # Initially not occupied
        self.E_free_0 = 0  # Initial energy of kinetic electrons, Initally zero
        self.rho_j_0  = self.m_j * self.FermiSolver.fermi(temperature, 0) # occupied acording to initial temperature

        ## derived from these
        #self.R_VB_0 = np.sum(self.rho_j_0)  # Initially occupied up to Fermi Energy
        self.T_0 = temperature  # Initial thermal energy of the average valence electron

        # This vector contains all parameters that are tracked over time
        self.states_per_voxel = 3 + self.N_j  # Just the number of entries for later convenience
        self.state_vector_0 = np.array([
                                   self.R_core_0,
                                   self.R_free_0,
                                   self.E_free_0,
                                   *self.rho_j_0] * self.Nsteps_z).reshape(self.Nsteps_z, self.states_per_voxel)

        # This is a constant matrix needed in rate_free()
        self.energy_differences = np.zeros((self.Nsteps_z, self.N_j, self.N_photens))
        for i in range(self.N_photens):
            self.energy_differences[:,:,i] = (self.E_j - self.E_i[i])

    def get_resonant_states(self, Emin, Emax, npoints=10):
        """
        Returns the number of states resonant to the photon energy E,
        assuming a certain resonant width.
        The DoS data stored in the file should be normalized to unity.
        A re-sampling is done to prevent any funny businness due to the sampling density of the DoS.
        """

        X = np.linspace(Emin, Emax, npoints)
        Y = np.interp(X, self.E_j, self.DoS)
        return np.trapz(y=Y, x=X)

    def pulse_profiles(self, t):
        """
        For now this returns a simple Gaussian profile for each photon energy Ej.
        A call costs 9.7 µs on jupyterhub for two Energies at one time - perhaps this can be
        reduced by using interpolation between a vector in the workspace.
        """
        result = self.I0.copy()/self.atomic_density # normalize from photons/nm² to (photons nm)/ atom
        result[~self.resonant] = 0
        result[self.resonant] *= np.exp(-0.5 * ((t - self.t0[self.resonant]) / self.tdur_sig[self.resonant]) ** 2)\
                                 * 1 / (np.sqrt(2 * np.pi) * self.tdur_sig[self.resonant])
        return result

    def make_valence_energy_axis(self, N_j: int, min=-6, finemax=15, max=50):
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

            def fill_biggest_gap(pointlist):
                """
                This function takes a list of points and appends a point in the middle of the biggest gap
                """
                pointlist = np.array(np.sort(pointlist))
                gaps = pointlist[1:]-pointlist[:-1]
                biggest_gap_index = np.argsort(gaps)[-1]
                biggest_gap = gaps[biggest_gap_index]
                list_before = pointlist[:biggest_gap_index+1]
                new_value = pointlist[biggest_gap_index] + 0.5*biggest_gap
                list_after = pointlist[biggest_gap_index+1:]
                return np.concatenate((list_before, [new_value,], list_after))

            # The energies E_i and 0 must be in the axis
            enax_j_fine = [min, 0, finemax]+list(self.E_i[self.E_i<=finemax])
            # Fill up the gaps
            while len(enax_j_fine)<N_j_fine:
                enax_j_fine = fill_biggest_gap(enax_j_fine)

            dE = np.mean(enax_j_fine[1:]-enax_j_fine[:-1])
            #The same for the coarse part
            enax_j_coarse = [finemax+dE, max]+list(self.E_i[self.E_i>finemax])
            while len(enax_j_coarse)<N_j_coarse:
                enax_j_coarse = fill_biggest_gap(enax_j_coarse)

            enax_j = np.concatenate((enax_j_fine, enax_j_coarse))

            if not len(enax_j) == N_j:
                warnings.warn(
                    'Energy Axis turned out longer or shorter than planned. What went wrong?')
                self.N_j = len(enax_j)

            def edgepoints(middles):
                """ Opposite of midpoints """
                edges = np.empty(middles.shape[0] + 1)
                edges[1:-1] = (middles[1:] + middles[:-1]) / 2
                edges[0] = middles[0] - (middles[1] - middles[0]) / 2
                edges[-1] = middles[-1] + (middles[-1] - middles[-2]) / 2
                return edges

            return enax_j, edgepoints(enax_j)


class FermiSolver:
    """
    This class finds the target "thermal equilibrium" electron distribution in the valence band for any given distribution or a set of inner energy and population
    """

    def __init__(self, par, m_j, DEBUG=False):
        self.par0 = lmfit.Parameters()
        self.par0.add('T', value=350, min=300, max=1e6)
        self.par0.add('Ef', value=1, min=-9, max=20)
        self.par = par
        self.enax = par.E_j
        self.m_j = m_j
        self.kB = 8.617333262145e-5
        self.DEBUG = DEBUG

    def fermi(self, T: float, Ef: float):
        # Due to the exponential I get a floating point underflow when calculating the fermi distribution naively, hence the extra effort
        energy_ratios = (self.enax - Ef) / (T * self.kB)
        fermi_distr = np.zeros(energy_ratios.shape)
        calculatable = np.abs(energy_ratios) < 15
        fermi_distr[calculatable] = 1 / (np.exp(energy_ratios[calculatable]) + 1)
        fermi_distr[energy_ratios < -15] = 1
        fermi_distr[energy_ratios > 15] = 0
        # if self.DEBUG:
        #    check_bounds(fermi_distr, message='Fermi distribution in fermi()')
        return fermi_distr

    def inner_energy(self, occupation):
        return np.sum(self.enax * occupation)  # np.trapz(self.enax * occupation, self.enax)

    def optimizable(self, pars, U_target, R_target):
        """
        This is a loss function to minimize to find a combination of Fermi energy and temperature for
        a given inner energy and valence band occupation.
        """
        vals = pars.valuesdict()
        T = vals['T']
        Ef = vals['Ef']
        occ = self.m_j * self.fermi(T, Ef)
        U = self.inner_energy(occ)
        R = np.sum(occ)
        ur = np.abs(U - U_target)
        rr = np.abs(R - R_target)
        # print(f'Resiuals rU:{ur}, rR{rr}')
        return ur, rr

    def solve(self, U, R):
        # print(f'Looking for a solution for U: {U}, R:{R}')
        res = lmfit.minimize(self.optimizable, self.par0, args=(U, R), method='nelder')  # powell
        ## Check for large residuals
        res_U, res_R = res.residual
        if res_U > 0.1:
            if self.DEBUG: print(f'Residual in U remained significant ({res_U}) Setting to NaN')
            return np.nan, np.nan
        if res_R > 0.1:
            if self.DEBUG: print(f'Residual in R remained significant ({res_R}) Setting to NaN')
            return np.nan, np.nan
        ## Check for running into parameter limits
        if res.params['T'].value < res.params['T'].min + 1e-6:
            if self.DEBUG: print('Minimum T reached!')
            return np.nan, np.nan
        if res.params['T'].value > res.params['T'].max - 1e-6:
            if self.DEBUG: print('Maximum T reached!')
            return np.nan, np.nan
        if res.params['Ef'].value < res.params['Ef'].min + 1e-6:
            if self.DEBUG: print('Minimum Ef reached!')
            return np.nan, np.nan
        if res.params['Ef'].value > res.params['Ef'].max - 1e-6:
            if self.DEBUG: print('Maximum Ef reached!')
            return np.nan, np.nan
        if not res.success:
            # raise RuntimeError('No solution found!')
            return np.nan, np.nan

        else:
            self.par0 = res.params  # pass solution for the next iteration
            return res.params['T'].value, res.params['Ef'].value

    def solve_target_distribution(self, momentary_distribution):
        """
        Calculates a target distribution for a specific incident distribution by optimization of Ef and T
        """
        U_is = self.inner_energy(momentary_distribution)
        R_is = np.sum(momentary_distribution)  # np.trapz(momentary_distribution, x=self.enax)

        T, Ef = self.solve(U_is, R_is)
        if self.DEBUG: print(U_is, R_is, T, Ef)
        return self.m_j * self.fermi(T, Ef)

    def load_lookup_tables(self):
        try:
            ld = np.load('./fermi_lookup_table.npz')
        except:
            raise OSError('Lookup table file not found.')
        self.Upoints = ld['Upoints']
        self.Rpoints = ld['Rpoints']
        self.Urange = ld['Urange']
        self.Rrange = ld['Rrange']
        self.precalc_temperatures_points = ld['temp_points']
        self.precalc_fermi_energies_points = ld['ferm_points']
        self.Rmin = np.min(self.Rrange)
        self.Rmax = np.max(self.Rrange)
        self.Umin = np.min(self.Urange)
        self.Umax = np.max(self.Urange)
        print('Loaded lookup table successfully.')

    def lookup_TEf_from_UR(self, U_is, R_is):
        if not (self.Umin-0.1 < U_is < self.Umax):
            raise ValueError(f'U: {U_is} out of bounds of lookup table')
        if not (self.Rmin < R_is < self.Rmax):
            raise ValueError(f'R: {R_is} out of bounds of lookup table')

        if self.DEBUG: print(U_is, R_is)

        T = sc.interpolate.griddata((self.Upoints, self.Rpoints), self.precalc_temperatures_points, (U_is, R_is),
                                    method='nearest')
        Ef = sc.interpolate.griddata((self.Upoints, self.Rpoints), self.precalc_fermi_energies_points, (U_is, R_is),
                                     method='nearest')

        if np.isnan(T) or np.isnan(Ef):
            print(f'Lookup failed. Trying direct solve for R={R_is:.1f} and U={U_is:.1f}')
            T, Ef = self.solve(U_is, R_is)  # try direct solving
            self.Upoints = np.append(self.Upoints, U_is)
            self.Rpoints = np.append(self.Rpoints, R_is)
            self.precalc_temperatures_points = np.append(self.precalc_temperatures_points, T)
            self.precalc_fermi_energies_points = np.append(self.precalc_fermi_energies_points, Ef)
            if np.isnan(T) or np.isnan(Ef):
                raise ValueError(f'Could not find a combination of Ef and T for R={R_is:.1f} and U={U_is:.1f}')
            print(f'Direct solve worked and lead to: T={T:.1f} and Ef={Ef:.1f}')
        return T, Ef

    def lookup_target_distribution(self, momentary_distribution):
        """
        Calculates a target distribution for a specific incident distribution by optimization of Ef and T
        """
        U_is = self.inner_energy(momentary_distribution)
        R_is = np.sum(momentary_distribution)  # np.trapz(momentary_distribution, x=self.enax)

        T, Ef = self.save_lookup_TEf_from_UR(U_is, R_is)

        if self.DEBUG: print(U_is, R_is, T, Ef)
        return self.m_j * self.fermi(T, Ef)

    def plot_lookup_tables(self):
        Ugrid, Rgrid = self.Ugrid, self.Rgrid  # np.meshgrid(self.Urange, self.Rrange)
        temperatures_gr = sc.interpolate.griddata((self.Upoints, self.Rpoints),
                                                  self.precalc_temperatures_points, (Ugrid, Rgrid), method='nearest')
        fermi_energies_gr = sc.interpolate.griddata((self.Upoints, self.Rpoints),
                                                    self.precalc_fermi_energies_points, (Ugrid, Rgrid),
                                                    method='nearest')

        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(1, 2, 1)

        pl1 = ax1.pcolormesh(Ugrid, Rgrid, temperatures_gr, cmap=plt.cm.coolwarm,
                             norm=mpl.colors.LogNorm(vmin=200, vmax=1e6),
                             linewidth=1,shading = 'nearest')
        fig.colorbar(pl1)  # , shrink=0.5, aspect=5
        ax1.set_title('Electron temperature (K)')
        ax1.set_ylabel('Valence electrons per atom')
        ax1.set_xlabel('Inner energy per atom (eV)')

        ax2 = fig.add_subplot(1, 2, 2)

        pl2 = ax2.pcolormesh(Ugrid, Rgrid, fermi_energies_gr, cmap=plt.cm.seismic, vmin=-20, vmax=20,
                             linewidth=1,shading = 'nearest')
        fig.colorbar(pl2)  # , shrink=0.5, aspect=5
        ax2.set_title('Fermi energy shift (eV)')
        ax2.set_ylabel('Valence electrons per atom')
        ax2.set_xlabel('Inner energy per atom (eV)')
        plt.tight_layout()

        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(1, 2, 1)

        pl1 = ax1.pcolormesh(temperatures_gr, fermi_energies_gr, self.Ugrid, \
                             cmap=plt.cm.coolwarm,
                             linewidth=1,shading = 'nearest')
        fig.colorbar(pl1)  # , shrink=0.5, aspect=5
        ax1.set_title('Inner Energy')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Fermi Energy')

        ax2 = fig.add_subplot(1, 2, 2)

        pl2 = ax2.pcolormesh(temperatures_gr, fermi_energies_gr, self.Rgrid, cmap=plt.cm.seismic,
                             linewidth=1,shading = 'nearest')
        fig.colorbar(pl2)  # , shrink=0.5, aspect=5
        ax2.set_title('Population')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Fermi Energy')
        plt.tight_layout()
        plt.pause(0.1)
        plt.show(block=False)

    def generate_lookup_tables(self, N=100, save=True):

        assert np.mod(N, 2) == 0
        temperatures = np.logspace(0, 6, N - 1) + 300
        fermis = np.logspace(-3, 1.5, int(N / 2))
        fermis = np.concatenate((-fermis[::-1], fermis[1:]))
        print(
            f'Starting to generate lookup tables for T between {np.min(temperatures):.1f} to {np.max(temperatures):.1f} and Ef between {np.min(fermis):.1f} and {np.max(fermis):.1f}')

        Tgrid, Efgrid = np.meshgrid(temperatures, fermis)

        self.Ugrid = np.empty((N - 1, N - 1))
        self.Rgrid = np.empty((N - 1, N - 1))

        for iT, T in enumerate(temperatures):
            for iF, Ef in enumerate(fermis):
                distr = self.m_j * self.fermi(T, Ef)
                self.Ugrid[iT, iF] = self.inner_energy(distr)
                self.Rgrid[iT, iF] = np.sum(distr)
                # print(T,Ef, '->', self.Ugrid[iT, iF], self.Rgrid[iT, iF])

        self.Upoints = self.Ugrid.T.flatten()
        self.Rpoints = self.Rgrid.T.flatten()
        self.precalc_temperatures_points = Tgrid.flatten()
        self.precalc_fermi_energies_points = Efgrid.flatten()
        # self.Urange = self.Ugrid[:,0]
        # self.Rrange = self.Rgrid[0]
        print('Lookup tables generated.')
        if save:
            savename = 'fermi_lookup_table.npz'
            print(f'Saving at ./{savename}')
            np.savez(savename, Rpoints=self.Rpoints, Upoints=self.Upoints,
                     Rgrid=self.Rgrid, Ugrid=self.Ugrid,
                     temp_points=self.precalc_temperatures_points,
                     ferm_points=self.precalc_fermi_energies_points)
        self.Rmin = np.min(self.Rgrid)
        self.Rmax = np.max(self.Rgrid)
        self.Umin = np.min(self.Ugrid)
        self.Umax = np.max(self.Ugrid)

    def save_lookup_TEf_from_UR(self, U, R):
        T, Ef = self.lookup_TEf_from_UR(U, R)
        self.par0['T'].value = T
        self.par0['Ef'].value = Ef
        T, Ef = self.solve(U, R)
        return T, Ef

## Main Simulation

class XNLsim:
    def __init__(self, par, DEBUG=False, load_tables=True):
        self.DEBUG = DEBUG
        self.intermediate_plots = False
        self.par = par
        self.par.make_derived_params(self)


        if load_tables:
            try:
                self.par.FermiSolver.load_lookup_tables()
            except OSError:
                while True:
                    YN = input('Could not load table. Generate new ones ? (Y/N)')
                    if YN in ['y','Y']:
                        self.par.FermiSolver.generate_lookup_tables()
                        break
                    elif YN in ['n','N']:
                        break
                    else:
                        print('Invalid answer.')
        else:
            self.par.FermiSolver.generate_lookup_tables()

        # initiate storing intermediate results
        self.call_counter = 0
        self.thermal_occupations = None

        # Pre-initialization for efficient memory usage
        self.res_inter      = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.nonres_inter   = np.empty((self.par.Nsteps_z, self.par.N_j, self.par.N_photens), dtype = np.float64)
        self.ch_decay       = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.el_therm       = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.el_scatt       = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.mean_free      = np.empty((self.par.Nsteps_z), dtype = np.float64)
        self.mean_valence   = np.empty((self.par.Nsteps_z), dtype = np.float64)
        self.state_vector   = np.empty(self.par.state_vector_0.shape, dtype = np.float64)
        self.T              = self.par.T_0
        self.target_distributions = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
    ###################
    ### Processes
    ###################

    def run(self, t_span, method, rtol, atol, plot=False, return_full_solution=False):
        """
        Run the simulation once for a z-stack

        :param t_span: Time axis to simulate within, limits in femtoseconds
        :param method: Use 'RK45' for or 'DOP853' (tha latter for very small error goals)
        :param rtol: Relative error goal
        :param atol: Absolute error goal
        :param plot: Boolean to plot results
        :return: incident_pulse_energies, transmitted_pulse_energies
        """
        global RTOL
        RTOL = rtol
        ### Solve Main problem
        sol = solve_ivp(self.time_derivative, t_span=t_span, \
                        dense_output=True, y0=self.par.state_vector_0.flatten(), method=method, rtol=rtol,
                        atol=atol)  # DOP853 or RK45

        soly = sol.y.reshape((self.par.Nsteps_z, self.par.states_per_voxel, len(sol.t)))

        ### Since they weren't saved, calculate transmission again
        sol_photon_densities = np.zeros((self.par.Nsteps_z, self.par.N_photens, len(sol.t)))
        for it, t in enumerate(sol.t):
            sol_photon_densities[:, :, it] = self.z_dependence(t, soly[:, :, it])[:,self.par.resonant]

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
            E_j = self.par.E_j

        # Due to the exponential I get a floating point underflow when calculating the fermi distribution naively, hence the extra effort
        energy_ratios = np.outer(E_j, 1/(self.par.kB * T)) #E_j / T
        fermi_distr = np.zeros(energy_ratios.shape)
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
        core_occupation = np.outer((R_core / self.par.M_core), np.ones(self.par.N_j))
        valence_occupation = rho_j/self.par.m_j # relative to the states at that energy
        gs_intensity = np.zeros(valence_occupation.shape)
        gs_intensity[:,self.par.resonant] =  N_Ej[:,self.par.resonant] / self.par.lambda_res_Ei
        if self.DEBUG:
            check_bounds(core_occupation, message='Valence occupation in proc_res_inter_Ej()')
            check_bounds(valence_occupation, message='Valence occupation in proc_res_inter_Ej()')
        return (core_occupation - valence_occupation) * gs_intensity # returns j,i

    # Nonresonant interaction
    def proc_nonres_inter(self, N_Ej, rho_j):
        valence_occupation = rho_j/self.par.R_VB_0 # relative to the number valence states in the ground state
        if self.DEBUG: check_bounds(valence_occupation, message='valence occupation deviation in proc_nonres_inter()')
        #TODO: Somehow avoid this slow loop
        result = np.empty((self.par.Nsteps_z,self.par.N_j, self.par.N_photens))
        for iz in range(self.par.Nsteps_z):
            result[iz] = np.outer(valence_occupation[iz], N_Ej[iz, self.par.resonant])
        return result / self.par.lambda_nonres # returns z, j,i

    # Core-hole decay
    def proc_ch_decay(self, R_core, rho_j):
        core_holes = (self.par.M_core - R_core) # z
        R_VB = np.sum(rho_j,axis=1)
        valence_resonant_occupation = rho_j/(self.par.m_j) #rho_j_0 # z, j
        valence_resonant_occupation_share = (valence_resonant_occupation.T/R_VB).T
        valence_total_occupation_change = R_VB/self.par.R_VB_0
        return (core_holes.T * valence_resonant_occupation_share.T * valence_total_occupation_change).T / self.par.tau_CH
    
    # Electron Thermalization
    def proc_el_therm(self, rho_j, r_j):
        el_therm = (r_j - rho_j) / self.par.tau_th
        sum_of_changes = np.sum(el_therm, 1) # This must be zero but can deviate due to numerics
        if self.DEBUG:
            global RTOL
            if np.any(sum_of_changes>RTOL):
                warnings.warn('Correcting a significant non-zero sum in thermalization')
        correction = np.abs(el_therm)* np.outer(sum_of_changes/np.sum(np.abs(el_therm),1),np.ones(self.par.N_j))
        el_therm = el_therm - correction # I simply subtract the average deviation from all
        return el_therm

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
    def mean_valence_energy(self, rho_j):
        """
        I am not sure yet which electron's energy contributes. My instinct says only valence,
        Martin says also free AND core. Here is valence and Free. Check!
        :param rho_j:
        :return:
        """
        total_energy = np.sum(self.par.E_j * rho_j, axis=1)
        return total_energy / (np.sum(rho_j, axis=1))


    # unpacks state vector and calls all the process functions
    def calc_processes(self, N_Ej, states, r_j):
        """
        Calculates all the processes for all z
        dimeninality of each j-resolved variable is [iz,j]
        """

        R_core = states[:, 0]
        R_free = states[:, 1]
        E_free = states[:, 2]
        rho_j = states[:, 3:]
        
        #TODO: After thinking about it, an explicit return might be better readible AND faster, since I initialize the variables in the functions
        self.res_inter = self.proc_res_inter_Ej(N_Ej, R_core, rho_j)
        self.nonres_inter = self.proc_nonres_inter(N_Ej, rho_j)
        self.ch_decay = self.proc_ch_decay(R_core, rho_j)
        self.el_therm = self.proc_el_therm(rho_j, r_j)
        self.el_scatt = self.proc_free_scatt(R_free)
        self.mean_free = self.mean_free_el_energy(R_free, E_free)
        self.mean_valence = self.mean_valence_energy(rho_j)
        #return res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence

    def rate_N_dz_j_direct(self, N_Ej, states):
        """
        Calculates only dN/dz for a given z.
        This one is for the directly coded light propagation
        """
        R_core, R_free, E_free = states[0:3]
        rho_j = states[3:]
        core_occupation = (R_core / self.par.M_core)
        valence_occupation = rho_j/self.par.m_j # relative to the states at that energy
        return - (core_occupation.T - valence_occupation.T).T * (N_Ej * self.par.lambda_res_Ej_inverse)

    ############################
    ### Rates - time derivatives
    ############################
    def rate_j(self):
        global RTOL
        rho_j = self.state_vector[:, 3:]
        R_VB = np.sum(rho_j, axis=1)
        direct_augers = self.ch_decay
        indirect_augers = ((rho_j * np.outer(np.sum(self.ch_decay, axis = 1),np.ones(self.par.N_j))).T/R_VB).T
        holes_j = self.par.m_j - rho_j
        if np.any(holes_j<0):
            mn = np.min(holes_j)
            if mn<-RTOL:
                warnings.warn(f'negative electron hole density found down to: {mn}')
            holes_j[holes_j<0]=0
        holes = self.par.M_VB - np.sum(rho_j,1) # the sum over j of holes_j/holes has to be 1
        if np.any(holes < 1e-10):
            #TODO: Check why this triggers often in the very first time step
            warnings.warn(f'Number of holes got critically low for computational accuracy.')
            holes_j[holes_j<1e-10] *= 0

        return self.res_inter - np.sum(self.nonres_inter, axis = 2) - direct_augers -\
               indirect_augers + self.el_therm + ((holes_j.T/holes)* self.el_scatt).T

    def rate_core(self):
        return np.sum(self.ch_decay, axis=1) - np.sum(self.res_inter, axis=1)

    def rate_free(self):
        return np.sum(self.nonres_inter, axis=(1,2)) + np.sum(self.ch_decay, axis = 1) - self.el_scatt

    def rate_E_free(self):
        # , nonres_inter, ch_decay, el_scatt, mean_free, mean_valence):

        #nonres_inter = self.nonres_inter
        #ch_decay = self.ch_decay
        #scatt = self.el_scatt

        R_free = self.state_vector[:, 1]
        E_free = self.state_vector[:, 2]

        result = np.zeros(self.par.Nsteps_z)
        interesting = R_free > 0
        result[interesting] = np.sum(self.nonres_inter[interesting] * self.par.energy_differences[interesting], axis = (1,2))\
               + np.sum(self.ch_decay * self.par.E_j, axis = 1)[interesting]\
               - self.el_scatt[interesting]*E_free[interesting]/R_free[interesting]
        return result


    ###################
    ### Propagation of photons through the sample
    ###################

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

        # Z-loop for every photon energy
        N_Ej_z[1, :] = zstep_euler(self, N_Ej_z[0, :], state_vector, 0)  # First step with euler
        for iz in range(2, self.par.Nsteps_z):
            N_Ej_z[iz, :] = double_zstep_RK(self, N_Ej_z[iz - 2, :], state_vector, iz - 2)

        return N_Ej_z

    ##########################################################
    ### Main differential describing the time evolution of voxels
    ##########################################################

    def time_derivative(self, t, state_vector_flat):
        if self.DEBUG: print('t: ', t)
        # Reshape the state vector into sensible dimension
        self.state_vector = state_vector_flat.reshape(self.par.Nsteps_z, self.par.states_per_voxel)
        check_bounds(self.state_vector[:, 3], 0, np.inf,
                     message='Temperature in time_derivative.')  # The temperature must never become negative

        # Determine thermalized distributions
        for iz, z in enumerate(self.par.zaxis):
            current_rho_j = self.state_vector[iz,3:]
            R = np.sum(current_rho_j)
            U = np.sum(current_rho_j*self.par.E_j)#np.trapz(current_rho_j*self.par.E_j, x = self.par.E_j)
            if iz < 2:
                # When jumping to the surfacem do the safer lookup.
                T, E_f = self.par.FermiSolver.save_lookup_TEf_from_UR(U,R)
                if self.DEBUG and (iz == 0):
                    print(U,R,'->',T, E_f)
            else:
                # From then on use the last results as inputs for the solver.
                T, E_f = self.par.FermiSolver.solve(U,R)
            self.target_distributions[iz, :] = self.par.FermiSolver.fermi(T, E_f) * self.par.m_j


        # Calculate photon transmission as save it
        N_Ej_z = self.z_dependence(t, self.state_vector)

        #This calculates (and writs to self.):
        # res_inter, nonres_inter, ch_decay, el_therm, el_scatt, mean_free, mean_valence
        self.calc_processes(N_Ej_z[:, :], self.state_vector[:, :], self.target_distributions)
        
        
        derivatives = np.empty(self.state_vector.shape)
        derivatives[:, 0] = self.rate_core()
        derivatives[:, 1] = self.rate_free()
        derivatives[:, 2] = self.rate_E_free()
        derivatives[:, 3:] = self.rate_j()

        # Debug plotting
        if self.intermediate_plots:
            if np.mod(self.call_counter, 20) == 0:
                self.plot_z_dependence(N_Ej_z)
                self.plot_occupancies(self.state_vector)
                self.plot_derivatives(derivatives)
                plt.show(block=False)
                plt.pause(0.1)

        self.call_counter += 1
        return derivatives.flatten()

    ##########################################################
    ### Auxiliary plotting functions
    ##########################################################
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
        plt.plot(self.par.zaxis, (state_vector[:, 2] - self.par.R_VB_0) / self.par.M_VB,
                 label='VB occupation variation')
        plt.plot(self.par.zaxis, state_vector[:, 3] - 300, label='T')
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


    def plot_results(self, sol, sol_photon_densities):
        PAR = self.par
        ## Plotting
        soly = sol.y.reshape((PAR.Nsteps_z, PAR.states_per_voxel, len(sol.t)))
        sol.core = soly[:, 0, :]
        sol.R_free = soly[:, 1, :]
        sol.E_free = soly[:, 2, :]
        sol.rho_j = soly[:, 3:, :]
        sol.R_VB = np.sum(sol.rho_j,1)
        
        sol.photon_densities = sol_photon_densities
        
        ### Also reconstruct the temperatures and Fermi energies
        sol.temperatures = np.zeros((len(sol.t),self.par.Nsteps_z))
        sol.fermi_energies = np.zeros((len(sol.t),self.par.Nsteps_z))
        
        for it,t in enumerate(sol.t):
            for iz in range(self.par.Nsteps_z):
                U = np.sum(sol.rho_j[iz,:,it]*self.par.E_j)
                R = sol.R_VB[iz,it]
                T, Ef = self.par.FermiSolver.save_lookup_TEf_from_UR(U, R)
                if self.DEBUG and (iz==0):
                    print(U,R,'->',T, Ef)
                sol.temperatures[it,iz], sol.fermi_energies[it,iz] = (T, Ef)
            
            
        fig, axes = plt.subplots(3, 2, figsize=(8, 8))
        plt.sca(axes[0, 0])
        plt.title('State occupation changes')
        plt.pcolormesh(sol.t, PAR.E_j +PAR.E_f,
                       (sol.rho_j[0])/np.outer(PAR.m_j,np.ones(sol.t.shape)),
                       cmap = plt.cm.seismic, vmin = -1, vmax = 1, shading = 'nearest')#-np.outer(PAR.rho_j_0,np.ones(sol.t.shape))
        plt.colorbar(label = 'Occupacion change')
        plt.xlabel('t (fs)')
        plt.ylabel('Energy (eV)')
        plt.title('Surface layer Valence occupation changes')


        plt.sca(axes[0, 1])
        plt.title('Kinetic electrons')

        plt.plot(sol.t, np.mean(sol.R_free, 0), label='Kinetic electrons')
        plt.plot(sol.t, sol.R_free[0], label='Surface')

        plt.ylabel('Occupation')
        plt.xlabel('t (fs)')
        plt.legend()

        plt.sca(axes[1, 0])
        plt.title('Energies Averaged over z')
        plt.plot(sol.t, np.mean(sol.temperatures, 1),'C0', label='Temperature')
        plt.plot(sol.t, sol.temperatures[:,0],'C1', label='Temperature @ Surface')
        plt.ylabel('T (K)')
        plt.legend()

        axcp = axes[1,0].twinx()
        plt.plot(sol.t, np.mean(sol.fermi_energies, 1),'C2', label='Fermi level shift')
        plt.plot(sol.t, sol.E_free[0],'C3', label='Fermi level shift @surface')
        plt.xlabel('t (fs)')
        plt.ylabel('E (eV)')

        plt.legend()

        plt.sca(axes[1, 1])
        plt.title('Photons')
        plt.xlabel('t (fs)')
        cols = plt.cm.cool(np.linspace(0,1,PAR.N_photens))
        for iE,E in enumerate(self.par.E_i):
            plt.plot(sol.t, sol_photon_densities[0, iE, :].T, c = cols[iE],ls = ':',
                 label=f'Incident {PAR.E_i[iE]:.2f} eV,  {PAR.lambda_res_Ei[iE]:.2f} nm' )
            plt.plot(sol.t, sol_photon_densities[-1, iE, :].T,c = cols[iE],
                 label=f'Transmitted' )

        #plt.plot(sol.t, sol_photon_densities[-1, :, :].T,
        #         label=[f'Transmitted {PAR.E_i[i]:.0f} eV,  {PAR.lambda_res_Ei[i]:.2f} nm' for i in
        #                range(PAR.N_photens)])

        #plt.plot(sol.t, sol_photon_densities[0, :, :].T,
        #         label=[f'Incident {PAR.E_i[i]:.0f} eV,  {PAR.lambda_res_Ei[i]:.2f} nm' for i in range(PAR.N_photens)])
        plt.legend()
        plt.ylabel('T')
        plt.xlabel('t (fs)')

        
        plt.sca(axes[2, 0])
        plt.title('Key populations at sample surface')
        plt.plot(sol.t,sol.core[0]/self.par.M_core, label = 'Core holes')
        plt.plot(sol.t,(sol.R_VB[0])/self.par.M_VB, label = 'Total Valence')
        cols = plt.cm.cool(np.linspace(0,1,PAR.N_photens))
        for iE,E in enumerate(self.par.E_i):
            plt.plot(sol.t,sol.rho_j[0,PAR.resonant,:][iE].T/self.par.m_j[PAR.resonant][iE],c = cols[iE], label = f'rho at {E:.2f}eV')
        plt.legend()

        plt.sca(axes[2, 1])
        plt.title('Total number of electrons (<z>)')
        plt.plot(sol.t,np.mean(sol.core + sol.R_free +sol.R_VB,0))
        plt.ylabel('Electrons')
        plt.xlabel('time (fs)')
        plt.ylim(0,None)

        plt.tight_layout()
        plt.show()
