import matplotlib.pyplot as plt
#import matplotlib as mpl
import scipy as sc
from scipy.integrate import solve_ivp
import numpy as np
import warnings

# Import all the parameters defined in the params file and processed in process_params
from .params import *

np.errstate(divide='raise')  # Raise division by zero as error if nominator is not also zero

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
        self.hbar = 6.582119569e-16  # eV s
        self.echarge = 1.602176634e-19  # J/eV

        ## Here I just "package" the variables from the params file into class attributes
        self.Nsteps_z = Nsteps_z  # Steps in Z
        self.N_photens = N_photens  # Number of distict resonant photon energies

        self.N_j = N_j

        self.timestep_min = timestep_min

        ## Sample thickness
        self.Z = Z  # nm

        self.atomic_density = atomic_density  # atoms per nm³
        self.photon_bandwidth = photon_bandwidth  # The energy span of the valence band that each resonant energy interacts with./ eV
        self.temperature = temperature  # Kelvin
        #self.work_function = work_function
        self.DoS_band_origin = DoS_band_origin
        self.DoS_band_dd_end = DoS_band_dd_end
        
        self.Energy_axis_min = Energy_axis_min
        self.Energy_axis_fine_until = Energy_axis_fine_until
        self.Energy_axis_max = Energy_axis_max
        

        ## Electronic state numbers per atom
        self.M_core = core_states
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
        self.mu_chem = 0
        self.T_0 = temperature  # Initial thermal energy of the average valence electron

        ## Incident photon profile - here with the dimension i in range(N_photens)
        self.I0_i = np.array(I0)
        self.t0_i = np.array(t0)
        self.tdur_sig_i = np.array(tdur_sig)
        self.E_i_abs = np.array(E_i)
        self.E_i = np.empty(N_photens)  # defined below
        

        assert (N_photens == len(I0) == len(t0) == len(tdur_sig) == len(E_i) == len(lambda_res_Ei)), \
            'Make sure all photon pulses get all parameters!'

    def make_derived_params(self):
        ## now some derived quantities
        self.zstepsize = self.Z / self.Nsteps_z
        self.zaxis = np.arange(0, self.Z, self.zstepsize)
        self.zedges = np.arange(0, self.Z+self.zstepsize, self.zstepsize)

        self.E_i = np.array(self.E_i_abs) - self.E_f  # self.E_i becomes relative to Fermi edge
        # Energy Axis
        self.E_j, self.enax_j_edges = self.make_valence_energy_axis(self.N_j)
        # Indizes where E_j == E_i
        self.resonant = np.array([(True if E_j in self.E_i else False) for E_j in self.E_j])

        ## Expanding the incident photon energies so that they match the tracked energies
        self.lambda_res_Ej = np.zeros(self.N_j, dtype=np.float64)
        self.I0 = np.zeros(self.N_j, dtype=np.float64)
        self.t0 = np.zeros(self.N_j, dtype=np.float64)
        self.tdur_sig = np.zeros(self.N_j, dtype=np.float64)
        self.lambda_res_Ej[self.resonant] = self.lambda_res_Ei
        self.lambda_res_Ej_inverse = np.zeros(self.lambda_res_Ej.shape, dtype=np.float64)
        self.lambda_res_Ej_inverse[self.resonant] = 1 / self.lambda_res_Ej[self.resonant]

        self.I0[self.resonant] = self.I0_i
        self.t0[self.resonant] = self.t0_i
        self.tdur_sig[self.resonant] = self.tdur_sig_i

        ## Load DoS data
        ld = np.load(self.DoS_shapefile)
        DoSdata = {}
        DoSdata['enax'] = ld[:, 0]
        DoSdata['DoS'] = ld[:, 1]
        # Dos_constant_from = 3  # inteprolate to general enaxis, but extend everything beyond +3eV to constant
        # DoSdata['DoS'][DoSdata['enax'] > Dos_constant_from] = DoSdata['DoS'][DoSdata['enax'] < Dos_constant_from][-1]
        # DoS_raw = np.interp(self.E_j, DoSdata['enax'][DoSdata['enax'] < Dos_constant_from],
        #                     DoSdata['DoS'][DoSdata['enax'] < Dos_constant_from])
        #
        #DoS_raw = np.interp(self.E_j, DoSdata['enax'], DoSdata['DoS'])
        def D_free(eV):
            joule = eV * self.echarge
            me = 9.1093837e-31  # Kg
            hbar = 1.054571817e-34  # Kg m /s^2
            V = 1e-27  # 1 nm^3 in m^3
            term1 = V / (self.atomic_density * 2 * np.pi ** 2)
            term2 = (2 * me / hbar ** 2) ** (3 / 2)
            with np.errstate(invalid='ignore'):
                res = term1 * term2 * np.sqrt(joule) * self.echarge  # Return DoS in states per atom and eV
            res[eV <= 0] = 0
            return res

        # The following makes sure that the states m_j correspond to the integral of the states they represent, even if the DoS varies within the given energy interval 
        f_DOS = sc.interpolate.interp1d(DoSdata['enax'],DoSdata['DoS'], kind = 'linear')
        def integrate_m(j, N_oversample = 30):
            low, high = self.enax_j_edges[j], self.enax_j_edges[j+1]
            if (low < np.min(DoSdata['enax'])) or (high > np.max(DoSdata['enax'])):
                return np.nan
            else:
                return np.trapz(f_DOS(np.linspace(low, high+1e-6,30)), x= np.linspace(low, high+1e-6,30))
        self.m_j = np.array([integrate_m(j) for j in range(self.N_j)])

        self.enax_dE_j = self.enax_j_edges[1:] - self.enax_j_edges[:-1]
        #self.m_j = DoS_raw * self.enax_dE_j  # scale with energy step size
        
        ## mj are normalized by the ground state population
        mj_up_to_incl_0 = self.m_j[self.E_j<0]#np.concatenate((self.m_j[self.E_j<0],[np.interp(0, self.E_j,self.m_j)]))
        occupied = np.sum(mj_up_to_incl_0) / np.nansum(self.m_j)  # part that is occupied in GS
        self.m_j = self.m_j / np.nansum(self.m_j)  # normalize to one to be sure
        self.m_j *= valence_GS_occupation / occupied  # scale to ground state occupation
        # FEG solution starts where DFT DoS stops
        self.m_j[self.E_j>self.DoS_band_dd_end] = D_free(self.E_j - self.DoS_band_origin)[self.E_j>self.DoS_band_dd_end]* \
                                                  self.enax_dE_j[self.E_j>self.DoS_band_dd_end]
        if np.any(self.m_j <=0) or np.any(np.isnan(self.m_j)):
            raise ValueError('Cannot work with zero or negative state densities!')

        self.M_VB = np.sum(self.m_j)

        self.FermiSolver = FermiSolver(self)

        ## Initial populations
        self.R_core_0 = self.M_core  # Initially fully occupied
        self.R_free_0 = 0  # Initially not occupied
        self.E_free_0 = 0  # Initial energy of kinetic electrons, Initally zero
        self.rho_j_0 = self.m_j * self.FermiSolver.fermi(temperature*self.kB, 0)  # occupied acording to initial temperature
        self.U_min = np.sum(self.E_j*self.m_j * self.FermiSolver.fermi(1*self.kB, 0))
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
            self.energy_differences[:, :, i] = self.E_j +(self.E_i[i] + self.E_f) #

    # def get_resonant_states(self, emin, emax, npoints=10):
    #     """
    #     Returns the number of states resonant to the photon energy E,
    #     assuming a certain resonant width.
    #     The DoS data stored in the file should be normalized to unity.
    #     A re-sampling is done to prevent any funny businness due to the sampling density of the DoS.
    #     """
    #
    #     X = np.linspace(emin, emax, npoints)
    #     Y = np.interp(X, self.E_j, self.DoS)
    #     return np.trapz(y=Y, x=X)

    def pulse_profiles(self, t):
        """
        For now this returns a simple Gaussian profile for each photon energy Ej.
        A call costs 9.7 µs on jupyterhub for two Energies at one time - perhaps this can be
        reduced by using interpolation between a vector in the workspace.
        """
        result = self.I0.copy()# / self.atomic_density  # normalize from photons/nm² to (photons nm)/ atom
        result[~self.resonant] = 0
        result[self.resonant] *= np.exp(-0.5 * ((t - self.t0[self.resonant]) / self.tdur_sig[self.resonant]) ** 2) \
                                 * 1 / (np.sqrt(2 * np.pi) * self.tdur_sig[self.resonant])
        return result

    def make_valence_energy_axis(self, N_j: int):
        """
            Creates an energy axis for the valence band, namely
                self.E_j
            and its edgepoints
                self.enax_j_edges
            Energies are relative to the fermi-level. 3/4 of all points fall into the range (min, finemax)
            Makes sure that the energies E_i correspond to a point in E_j and
            drops the closest points to keep the number N_j.
            :return:
            """
        # Unpack some values for frequent use
        min     = self.Energy_axis_min
        finemax = self.Energy_axis_fine_until
        max     = self.Energy_axis_max
        photon_bandwidth = self.photon_bandwidth
                
        
        N_j_fine = int(N_j * 3 / 4)
        N_j_coarse = int(N_j - N_j_fine)

        def fill_biggest_gap(pointlist, resonances: list):
            """
            This function takes a list of points and appends a point in the middle of the biggest gap
            """
            pointlist = np.array(pointlist)
            gaps = pointlist[1:] - pointlist[:-1]
            size_order = np.argsort(gaps)
            next_to_resonance = [((pointlist[ig] in resonances) or (pointlist[ig + 1] in resonances)) for ig, g in
                                 enumerate(gaps)]
            gaps[next_to_resonance] = 0
            biggest_gap_index = np.argsort(gaps)[-1]
            biggest_gap = gaps[biggest_gap_index]  # np.sort(gaps[~next_to_resonance])[-1]
            list_before = pointlist[:biggest_gap_index + 1]
            new_value = pointlist[biggest_gap_index] + 0.5 * biggest_gap
            list_after = pointlist[biggest_gap_index + 1:]
            return np.concatenate((list_before, [new_value, ], list_after))

        def set_initial_points(resonances: list, min: float, max: float, skip: list = []):
            """
            Enforce that specified energies have a size of by at least <photon_bandwith>
            unless they are too close together for that
            """
            points = list(np.sort([min, max] + skip + resonances))
            points_to_add = []
            for i, e in enumerate(points):
                # Skip first and last
                if (i == 0) or (e == max) or (e in skip):
                    continue
                De_to_last = e - points[i - 1]
                if De_to_last > 2 * photon_bandwidth:
                    points_to_add.append(e - photon_bandwidth)
                else:
                    print(
                        f'Energy {e:.2f} too close to others to satisfy the resonant bandwidth of {photon_bandwidth:.2f}')
                De_to_next = points[i + 1] - e
                if De_to_next > 2 * photon_bandwidth:
                    points_to_add.append(e + photon_bandwidth)
                else:
                    print(
                        f'Energy {e:.2f} too close to others to satisfy the resonant bandwidth of {photon_bandwidth:.2f}')
            return np.sort(np.append(points, points_to_add))

        # The energies E_i and 0 must be in the axis
        enax_j_fine = set_initial_points(list(self.E_i[self.E_i <= finemax]), min=min, max=finemax)#, skip=[0]
        # TODO: Can I avoid the necessity to include 0 so i can simulate resonances close to 0?

        # Fill up the gaps
        while len(enax_j_fine) < N_j_fine:
            enax_j_fine = fill_biggest_gap(enax_j_fine, list(self.E_i[self.E_i <= finemax]))

        dE = np.mean(enax_j_fine[1:] - enax_j_fine[:-1])
        # The same for the coarse part
        enax_j_coarse = set_initial_points(list(self.E_i[self.E_i > finemax]), min=finemax + dE, max=max)

        while len(enax_j_coarse) < N_j_coarse:
            enax_j_coarse = fill_biggest_gap(enax_j_coarse, list(self.E_i[self.E_i > finemax]))

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

    def plot_dos(self):
        plt.figure()
        plt.plot(self.E_j, self.m_j/self.enax_dE_j, '.-', label='m_j')
        plt.plot(self.E_j, self.rho_j_0/self.enax_dE_j, ':', label='rho_j_0')
        plt.axvline(0)
        plt.legend()
        plt.pause(0.1)


class FermiSolver:
    def __init__(self, par, DEBUG=False):
        self.DEBUG = DEBUG
        self.par = par
        self.par0 = 300. * self.par.kB, 0.

    def fermi(self, T: float, mu_chem: float):
        # Due to the exponential I get a floating point underflow when calculating the fermi distribution naively, hence the extra effort
        energy_ratios = (self.par.E_j - mu_chem) / (T)
        fermi_distr = np.zeros(energy_ratios.shape)
        not_too_small = energy_ratios > -200
        not_too_large = energy_ratios < 200
        calculatable =  not_too_small & not_too_large
        fermi_distr[calculatable] = 1 / (np.exp(energy_ratios[calculatable]) + 1)
        fermi_distr[~not_too_small] = 1
        fermi_distr[~not_too_large] = 0
        return fermi_distr

    def occupation(self, T, mu_chem):
        return self.fermi(T, mu_chem) * self.par.m_j

    def loss_fcn(self, X, U_target, R_target):
        T, mu_chem = X
        occ = self.occupation(T, mu_chem)
        U_is = np.sum(self.par.E_j * occ)
        R_is = np.sum(occ)
        return (U_is - U_target), (R_is - R_target)

    def loss_fcn_scalar(self, X, U_target, R_target):
        T, mu_chem = X
        occ = self.occupation(T, mu_chem)
        U_is = np.sum(self.par.E_j * occ)
        R_is = np.sum(occ)
        return (U_is - U_target)**2 + (R_is - R_target)**2

    def solve(self, U, R):
        U_max = np.sum(self.par.m_j*self.par.E_j*R/self.par.M_VB)
        if U<self.par.U_min:
            print('Too low of an energy demanded! ')#Setting to min!
            #U = self.par.U_min
            #return 300*self.par.kB, 0
        if U > U_max:
            print(f'Impossible Energy demanded: U={U:.1f}/{U_max:.1f} - result will not be precise!!!')
            U = U_max
            return np.inf, -1e3
            
        sol0 = sc.optimize.root(self.loss_fcn, self.par0, args=(U, R), method='lm', options={'xtol': RTOL *RTOL, 'maxiter': 400})
        err0 = np.max(np.abs(sol0.fun))
        sol = sol0
        err = err0

        if err0 > RTOL:
            bounds = [[290 * self.par.kB, sol0.x[0] * 2],
                      [sol0.x[1] - 100, sol0.x[1] + 100]]
            if bounds[0][0]>bounds[0][1]:
                bounds[0][1] = 1e8*self.par.kB                         # Up to 100 Million Degrees
            sol2 = sc.optimize.minimize(self.loss_fcn_scalar, sol0.x, args=(U, R),
                                       method = 'SLSQP',
                                       bounds = bounds,
                                       options = {'ftol':1e-10})
            err2 = np.max(np.abs(sol2.fun))
            if self.DEBUG: print('Second computation was needed')
            sol = sol2
            err = err2

        if err > RTOL:
            print(f'Still no good solution - Residual of {err}')

        T, mu_chem = sol.x


        if T < 0:
            print(U,R,'->',T, mu_chem)
            raise ValueError('Computed a negative temperature!')
        return T, mu_chem

## Main Simulation

class XNLsim:
    def __init__(self, par, DEBUG=False):
        self.DEBUG = DEBUG
        self.intermediate_plots = False
        self.par = par
        self.par.make_derived_params()


        # initiate storing intermediate results
        self.call_counter = 0
        self.thermal_occupations = None

        # Pre-initialization for efficient memory usage - no longer done but its a good overview
        """
        self.res_inter      = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.nonres_inter   = np.empty((self.par.Nsteps_z, self.par.N_j, self.par.N_photens), dtype = np.float64)
        self.ch_decay       = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.el_therm       = np.empty((self.par.Nsteps_z, self.par.N_j), dtype = np.float64)
        self.el_scatt       = np.empty((self.par.Nsteps_z), dtype = np.float64)
        #self.mean_free      = np.empty((self.par.Nsteps_z), dtype = np.float64)
        #self.mean_valence   = np.empty((self.par.Nsteps_z), dtype = np.float64)
        """
        self.state_vector   = np.empty(self.par.state_vector_0.shape, dtype = np.float64)
        #self.T              = self.par.T_0
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
        sol = solve_ivp(self.time_derivative, t_span=t_span,
                        dense_output=True, y0=self.par.state_vector_0.flatten(), method=method, rtol=rtol,
                        atol=atol, max_step=self.par.timestep_min)  # DOP853 or RK45

        soly = sol.y.reshape((self.par.Nsteps_z, self.par.states_per_voxel, len(sol.t)))

        ### Since they weren't saved, calculate transmission again
        sol_photon_densities = np.zeros((self.par.Nsteps_z+1, self.par.N_photens, len(sol.t)))
        for it, t in enumerate(sol.t):
            sol_photon_densities[:, :, it] = self.z_dependence(t, soly[:, :, it])[:, self.par.resonant]

        incident_pulse_energies = np.trapz(sol_photon_densities[0, :, :], x=sol.t)
        transmitted_pulse_energies = np.trapz(sol_photon_densities[-1, :, :], x=sol.t)

        if plot:
            self.plot_results(sol, sol_photon_densities)

        if return_full_solution:
            return incident_pulse_energies, transmitted_pulse_energies, sol
        else:
            return incident_pulse_energies, transmitted_pulse_energies

    # Resonant interaction
    def proc_res_inter_Ej(self, N_Ej, R_core, rho_j):
        core_occupation = np.outer((R_core / self.par.M_core), np.ones(self.par.N_j))
        valence_occupation = rho_j / self.par.m_j  # relative to the states at that energy
        gs_intensity = N_Ej * self.par.lambda_res_Ej_inverse
        if self.DEBUG:
            check_bounds(core_occupation, message='Valence occupation in proc_res_inter_Ej()')
            check_bounds(valence_occupation, message='Valence occupation in proc_res_inter_Ej()')
        return (core_occupation - valence_occupation) * gs_intensity  # returns j,i

    # Nonresonant interaction
    def proc_nonres_inter(self, N_Ej, rho_j):
        valence_change_to_GS = rho_j / self.par.R_VB_0  # relative to the number valence states in the ground state
        if self.DEBUG: check_bounds(valence_change_to_GS, message='valence occupation deviation in proc_nonres_inter()')
        # TODO: Somehow avoid this slow loop
        result = np.empty((self.par.Nsteps_z, self.par.N_j, self.par.N_photens))
        for iz in range(self.par.Nsteps_z):
            result[iz] = np.outer(valence_change_to_GS[iz], N_Ej[iz, self.par.resonant])
        ret = result / (self.par.lambda_nonres)
        if np.any(ret<0):
            if np.min(ret)<-RTOL:
                raise ValueError('Significant negative non-resonant decay!!')
            ret[ret<0] = 0

        #print(f'Finding nonres inter of {np.sum(ret,(1,2))} (normal)')
        return ret # returns z, j,i

    # Core-hole decay
    def proc_ch_decay(self, R_core, rho_j):
        core_holes_share = (self.par.M_core - R_core)  # z
        R_VB = np.sum(rho_j, axis=1)
        valence_resonant_occupation_share = rho_j / self.par.R_VB_0  # rho_j_0 # z, j
        valence_absolute_occupation_change = R_VB / self.par.R_VB_0
        return (core_holes_share.T * valence_resonant_occupation_share.T * valence_absolute_occupation_change).T / self.par.tau_CH
    # Electron Thermalization
    def proc_el_therm(self, rho_j, r_j):
        el_therm = (r_j - rho_j) / self.par.tau_th
        sum_of_changes = np.sum(el_therm, 1)  # This must be zero but can deviate due to numerics
        if self.DEBUG:
            global RTOL
            if np.any(sum_of_changes > RTOL):
                warnings.warn('Correcting a significant non-zero sum in thermalization')
        with np.errstate(invalid='ignore'):
            correction = np.abs(el_therm) * np.outer(sum_of_changes / np.sum(np.abs(el_therm), 1),
                                                     np.ones(self.par.N_j))
            np.nan_to_num(correction, copy=False, nan=0)
        el_therm = el_therm - correction  # I simply subtract the average deviation from all
        return el_therm

    # Free electron scattering
    def proc_free_scatt(self, R_free):
        return R_free / self.par.tau_free

    # Mean energy of kinetic electrons
    def mean_free_el_energy(self, R_free, E_free):
        empty = (R_free < 1e-9) # hardcoded precision limit of 1 neV
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
        #E_free = states[:, 2]
        rho_j = states[:, 3:]
        
        res_inter = self.proc_res_inter_Ej(N_Ej, R_core, rho_j)/self.par.atomic_density
        nonres_inter = self.proc_nonres_inter(N_Ej, rho_j)/self.par.atomic_density
        ch_decay = self.proc_ch_decay(R_core, rho_j)
        el_therm = self.proc_el_therm(rho_j, r_j)
        el_scatt = self.proc_free_scatt(R_free)
        #en_free = self.mean_free_el_energy(R_free, E_free)

        return res_inter, nonres_inter, ch_decay, el_therm, el_scatt#, en_free

    def rate_N_dz_j_direct(self, N_Ej, states):
        """
        Calculates only dN/dz for a given z.
        This one is for the directly coded light propagation
        """
        R_core, R_free, E_free = states[0:3]
        rho_j = states[3:]

        # Resonant
        core_occupation = (R_core / self.par.M_core)
        valence_occupation = rho_j / self.par.m_j  # relative to the states at that energy
        res_inter =  (core_occupation.T - valence_occupation.T).T * (N_Ej * self.par.lambda_res_Ej_inverse)

        # Non-resonant
        valence_change_to_GS = rho_j / self.par.R_VB_0  # relative to the number valence states in the ground state
        nonres_inter = np.outer(valence_change_to_GS, N_Ej) / (self.par.lambda_nonres)  # returns j,i
        return -res_inter - np.sum(nonres_inter,0)

    ############################
    ### Rates - time derivatives
    ############################
    def rate_j(self,res_inter,nonres_inter,ch_decay,el_therm,el_scatt):
        global RTOL
        rho_j = self.state_vector[:, 3:]
        R_VB = np.sum(rho_j, axis=1)
        direct_augers = ch_decay
        indirect_augers = ((rho_j * np.outer(np.sum(ch_decay, axis = 1),np.ones(self.par.N_j))).T/R_VB).T
        holes_j = self.par.m_j - rho_j
        if np.any(holes_j < 0):
            mn = np.min(holes_j)
            if mn < -RTOL:
                warnings.warn(f'Found negative electron hole density down to {mn}')
            #rho_j[rho_j>self.par.m_j] = self.par.m_j[rho_j>self.par.m_j]
            #holes_j = self.par.m_j - rho_j
        holes = np.sum(holes_j,1)  # the sum over j of holes_j/holes has to be 1
        if np.any(holes < 1e-10):
            warnings.warn(f'Number of holes got critically low for computational accuracy.')
            holes_j[holes_j < 1e-10] *= 0

        # This is the electrons which come back to the VB after they have scattered multiple times
        direct_scattering = ((holes_j.T / holes) * el_scatt).T
        energy_direct_scattering = np.sum((((holes_j.T / holes) * el_scatt).T) * self.par.E_j, 1) # This is how much energy they bring to the VB

        # all processes except scattering
        without_scattering = res_inter - np.sum(nonres_inter, axis=2) - direct_augers - \
                             indirect_augers + el_therm + direct_scattering

        # Check the energy balance (Only for specific debugging)
        # if False and self.DEBUG:
        #     energy_VB_before = np.sum(rho_j*self.par.E_j,1)
        #     energy_res_inter = np.sum(res_inter*self.par.E_j,1)
        #     energy_nonres_inter = np.sum(np.sum(nonres_inter, axis=2)*self.par.E_j,1)
        #     energy_direct_augers = np.sum(direct_augers*self.par.E_j,1)
        #     energy_indirect_augers = np.sum(indirect_augers*self.par.E_j,1)
        #     energy_all_augers = energy_indirect_augers+energy_direct_augers
        #     energy_el_therm = np.sum(el_therm*self.par.E_j,1)
        #     energy_direct_scattering = np.sum((((holes_j.T / holes) * el_scatt).T) * self.par.E_j,1)
        #     energy_change = energy_res_inter - energy_nonres_inter - energy_all_augers + energy_el_therm + energy_direct_scattering

        # Now calculate redistribution from scattering
        R_free = self.state_vector[:, 1]
        E_free = self.state_vector[:, 2]
        with np.errstate(invalid='ignore'):
            energy_incoming = (el_scatt * E_free / R_free) - energy_direct_scattering # This much energy is coming in
            np.nan_to_num(energy_incoming, copy=False, nan=0)  # Catching results of 0/0
            U_electrons = np.sum(rho_j * self.par.E_j, 1) / R_VB  # This much energy per el is in the electron distribution
            U_holes = np.sum(holes_j * self.par.E_j, 1) / holes  # This much energy per hole fits into the remaining holes
            electrons_to_move = energy_incoming / (-U_electrons + U_holes)
        scattering_contribution = (-rho_j.T * electrons_to_move / R_VB + 
                                  holes_j.T * electrons_to_move / holes).T


        if self.DEBUG:
            check_z_index = 1
            elc_error = np.sum(scattering_contribution, 1)[check_z_index]
            if elc_error>RTOL:
                print('Deviation from electron conservation: ', elc_error)
            should_be_new_energy = U_electrons[check_z_index]*R_VB[check_z_index] + energy_incoming[check_z_index]
            is_new_energy = np.sum((rho_j[check_z_index] + scattering_contribution[check_z_index]) * self.par.E_j)
            ec_error = np.abs(is_new_energy - should_be_new_energy)
            if ec_error>RTOL:
                print('Deviation from energy conservation: ', ec_error,' eV')

        return without_scattering + scattering_contribution

    def rate_core(self, res_inter, ch_decay):
        return np.sum(ch_decay, axis=1) - np.sum(res_inter, axis=1)

    def rate_free(self, nonres_inter, ch_decay, el_scatt):
        return np.sum(nonres_inter, axis=(1,2)) + np.sum(ch_decay, axis = 1) - el_scatt

    def rate_E_free(self, nonres_inter, ch_decay, el_scatt):

        R_free = self.state_vector[:, 1]
        E_free = self.state_vector[:, 2]

        result = np.zeros(self.par.Nsteps_z)
        with np.errstate(invalid='ignore'):
            scattering = el_scatt * E_free/R_free
            scattering[R_free<=0] = 0
        total_nonres = np.sum(nonres_inter * self.par.energy_differences, axis = (1,2))

        #ch_en_rate = np.sum(ch_decay * (self.par.E_f+self.par.E_j), axis = 1) #
        rho_j_new = self.state_vector[:, 3:]-ch_decay
        average_new_valence_en = self.par.E_f+np.sum(rho_j_new*self.par.E_j,1)/np.sum(rho_j_new,1)
        ch_en_rate = np.sum(ch_decay, axis = 1) * average_new_valence_en#

        result = total_nonres\
               + ch_en_rate\
               - scattering
        return result

    ###################
    ### Propagation of photons through the sample
    ###################

    def z_dependence(self, t, state_vector):
        def zstep_euler(self, N, state_vector, iz):
            res = N + self.rate_N_dz_j_direct(N, state_vector[iz, :]) * self.par.zstepsize# * self.par.atomic_density
            res[res<0] =0 #For large z-steps this can happen, but there cannot be a negative number of photons
            return res

        def double_zstep_RK(self, N, state_vector, iz):
            """
            Since I only know the states at specific points in z, I cheat by doubling the effective z step.
            """
            k1 = self.rate_N_dz_j_direct(N, state_vector[iz, :])
            k2 = self.rate_N_dz_j_direct(N + self.par.zstepsize * k1, state_vector[iz + 1, :])
            k3 = self.rate_N_dz_j_direct(N + self.par.zstepsize * k2, state_vector[iz + 1, :])
            k4 = self.rate_N_dz_j_direct(N + self.par.zstepsize * 2 * k3, state_vector[iz + 2, :])
            res = N + 0.3333333333333333 * self.par.zstepsize * (k1 + 2 * k2 + 2 * k3 + k4)
            res[res < 0] = 0  # For large z-steps this can happen, but there cannot be a negative number of photons
            return res

        # get current photon irradiation:
        N_Ej_z = np.zeros((self.par.Nsteps_z+1, self.par.N_j))

        N_Ej_z[0, :] = self.par.pulse_profiles(t)  # momentary photon densities at time t for each photon energy

        if self.DEBUG: print('Photons impinging per atom this timestep: ', N_Ej_z[0, self.par.resonant], 'i.e. ', N_Ej_z[0, self.par.resonant]/self.par.atomic_density ,'/atom')

        # Z-loop for every photon energy
        N_Ej_z[1, :] = zstep_euler(self, N_Ej_z[0, :], state_vector, 0)  # First step with euler
        for iz in range(2, self.par.Nsteps_z):
            N_Ej_z[iz, :] = double_zstep_RK(self, N_Ej_z[iz - 2, :], state_vector, iz - 2)
        N_Ej_z[-1, :] = zstep_euler(self, N_Ej_z[-2, :], state_vector, N_Ej_z.shape[0]-2)  # Last step back into vacuum with euler


        if np.any(N_Ej_z<0):
            print('wait a second!')
        return N_Ej_z
    
    def assert_positive_densities(self):
        ## Check for values that escape meaningful physics
        if np.any(self.state_vector<0):
            if np.any(self.state_vector<-RTOL):
                warnings.warn('Some states tried to become significantly negative!')
            self.state_vector[self.state_vector<0] = 0

    ##########################################################
    ### Main differential describing the time evolution of voxels
    ##########################################################
    def time_derivative(self, t, state_vector_flat):
        if self.DEBUG: print('t: ', t)

        global RTOL
        # Reshape the state vector into sensible dimension
        self.state_vector = state_vector_flat.reshape(self.par.Nsteps_z, self.par.states_per_voxel)

        # Option to enforce meaningful values - usually not necessary but
        self.assert_positive_densities()

        # Determine thermalized distributions
        for iz, z in enumerate(self.par.zaxis[:]):
            current_rho_j = self.state_vector[iz, 3:]

            R = np.sum(current_rho_j)
            U = np.sum(current_rho_j * self.par.E_j)  

            T, mu_chem = self.par.FermiSolver.solve(U, R)#

            # If on the surface, reset par0
            if iz < 1:
                self.par.FermiSolver.par0 = T, mu_chem

            if self.DEBUG and (iz == 0):
                    print(U, R, '->', T/self.par.kB, mu_chem)
            if np.isnan(T):
                print('!!')
                warnings.warn('Critical: Could not determine Temperature and Fermi Energy!')

            self.target_distributions[iz, :] = self.par.FermiSolver.fermi(T, mu_chem) * self.par.m_j

        # Calculate photon transmission as save it
        N_Ej_z = self.z_dependence(t, self.state_vector)

        res_inter, nonres_inter, ch_decay, el_therm, el_scatt = self.calc_processes(N_Ej_z[:-1, :], self.state_vector[:, :], self.target_distributions)
        
        
        derivatives = np.empty(self.state_vector.shape)
        derivatives[:, 0] = self.rate_core(res_inter, ch_decay)
        derivatives[:, 1] = self.rate_free(nonres_inter, ch_decay, el_scatt)
        derivatives[:, 2] = self.rate_E_free(nonres_inter, ch_decay, el_scatt)
        derivatives[:, 3:] = self.rate_j(res_inter,nonres_inter,ch_decay,el_therm,el_scatt)


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

        plt.plot(self.par.zedges, N_Ej_z[:,self.par.resonant]/N_Ej_z[0,self.par.resonant],'.-')
        plt.plot(self.par.zedges, np.exp(-((self.par.zedges/self.par.lambda_nonres)+(self.par.zedges/self.par.lambda_res_Ei[0]))), color = 'C3', label = 'Total GS')
        plt.plot(self.par.zedges, np.exp(-(self.par.zedges/self.par.lambda_nonres)), ls='--', label = 'Non-resonant GS')
        plt.plot(self.par.zedges, np.exp(-(self.par.zedges/self.par.lambda_res_Ei[0])), ls=':', label = 'Resonant GS')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('Transmission')
        self.axis_z.set_title(f'Z-Dependence')
        print(f'Transmission = {100* N_Ej_z[:,self.par.resonant][-1]/N_Ej_z[:,self.par.resonant][0]} %')

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
        plt.plot(self.par.zaxis, state_vector[:, 3] / self.par.kB - 300, label='T')
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
        sol.R_VB = np.sum(sol.rho_j, 1)

        sol.photon_densities = sol_photon_densities

        ### Also reconstruct the temperatures and Fermi energies
        sol.temperatures = np.zeros((len(sol.t), self.par.Nsteps_z))
        sol.chemical_potentials = np.zeros((len(sol.t), self.par.Nsteps_z))

        self.par.FermiSolver.par0 = 300 * self.par.kB,0

        sol.inner_energies = np.zeros((len(sol.t),self.par.Nsteps_z))
        for it, t in enumerate(sol.t):
            for iz in range(self.par.Nsteps_z):
                U = np.sum(sol.rho_j[iz, :, it] * self.par.E_j)
                R = sol.R_VB[iz, it]
                T, mu_chem = self.par.FermiSolver.solve(U, R)
                if iz ==0:
                    self.par.FermiSolver.par0 = T, mu_chem
                if np.isnan(T):
                    T, mu_chem = self.par.FermiSolver.save_lookup_Tmu_from_UR(U, R)
                # if self.DEBUG and (iz==0):
                #    print(U,R,'->',T, mu_chem)
                sol.temperatures[it, iz], sol.chemical_potentials[it, iz] = (T, mu_chem)
                sol.inner_energies[it,iz] = U # This is needed later to check the energy conservation

        fig, axes = plt.subplots(4, 2, figsize=(10, 8))
        plt.sca(axes[0, 0])
        plt.title('State occupation changes')
        plt.pcolormesh(sol.t, PAR.E_j + PAR.mu_chem,
                       (sol.rho_j[0] - np.outer(PAR.rho_j_0, np.ones(sol.t.shape))) / np.outer(PAR.m_j,
                                                                                               np.ones(sol.t.shape)),
                       cmap=plt.cm.seismic, vmin=-1, vmax=1, shading='nearest')  #
        plt.colorbar(label='Occupacion change')
        plt.xlabel('t (fs)')
        plt.ylabel('Energy (eV)')
        plt.title('Surface layer Valence occupation changes')

        plt.sca(axes[0, 1])
        plt.title('Kinetic electrons (surface)')
        plt.plot(sol.t, sol.R_free[0], c='C0')
        plt.ylabel('Number per atom', color='C0')
        plt.xlabel('t (fs)')
        ax012 = plt.twinx()
        ax012.plot(sol.t, sol.E_free[0], c='C1')
        plt.ylabel('Energy per atom', color='C1')

        plt.sca(axes[1, 0])
        plt.title('Energies Averaged over z')
        plt.plot(sol.t, sol.temperatures[:, 0], 'C0', label='Temperature (eV)')
        plt.plot(sol.t, sol.temperatures[:,1:], 'C0', lw = 0.5)

        plt.ylabel('T (eV)',color='C0')
        plt.legend(loc='upper left')

        axcp = axes[1, 0].twinx()
        plt.plot(sol.t, sol.chemical_potentials[:, 0], 'C1', label='Fermi level shift')
        plt.plot(sol.t, sol.chemical_potentials[:, 1:], 'C1', lw=0.5)
        plt.xlabel('t (fs)')
        plt.ylabel('mu (eV)',color='C1')

        plt.legend(loc='lower left')

        plt.sca(axes[1, 1])
        plt.title('Photons')
        plt.xlabel('t (fs)')
        cols = plt.cm.cool(np.linspace(0, 1, PAR.N_photens))
        for iE, E in enumerate(self.par.E_i):
            plt.plot(sol.t, sol_photon_densities[0, iE, :].T, c=cols[iE], ls=':',
                     label=f'Incident {PAR.E_i[iE]:.2f} eV,  {PAR.lambda_res_Ei[iE]:.2f} nm')
            plt.plot(sol.t, sol_photon_densities[-1, iE, :].T, c=cols[iE],
                     label=f'Transmitted')

        plt.legend()
        plt.ylabel('Photons / nm²')
        plt.xlabel('t (fs)')

        plt.sca(axes[2, 0])
        plt.title('Key populations at sample surface')
        plt.plot(sol.t, sol.core[0] / self.par.M_core, c='red', label='Core electrons')
        plt.plot(sol.t, (sol.R_VB[0]) / self.par.M_VB, c='green', label='Total valence')
        cols = plt.cm.cool(np.linspace(0, 1, PAR.N_photens))
        for iE, E in enumerate(self.par.E_i):
            plt.plot(sol.t, sol.rho_j[0, PAR.resonant, :][iE].T / self.par.m_j[PAR.resonant][iE], c=cols[iE],
                     label=f'rho at {E:.2f}eV')
        plt.legend()

        plt.sca(axes[2, 1])
        T = (sol.photon_densities[-1] / sol.photon_densities[0])  # /np.max(sol.photon_densities[0],1)
        for iE, E in enumerate(PAR.E_i):
            plt.plot(sol.t, T[iE], c=cols[iE], label=f'change at {E:.2f} eV')
        plt.axhline(np.exp(-self.par.Z/self.par.lambda_nonres), lw=0.3)
        plt.axhline(np.exp(-self.par.Z/self.par.lambda_res_Ei[0]), lw=0.3)
        plt.axhline(c='k', lw=0.3)
        plt.legend()
        plt.title('Transmitted / Incident photons')
        plt.xlabel('time (fs)')
        plt.ylabel('Rel. Transmission')


        plt.sca(axes[3, 0])
        plt.title('Electron Conservation (<z>)')
        electrons = np.mean(sol.core + sol.R_free +sol.R_VB,0)
        plt.plot(sol.t,electrons, label=f'Loss: {100*(electrons[-1]-electrons[0])/electrons[0]:.2f}%')
        plt.ylabel('No of Electrons')
        plt.xlabel('time (fs)')
        plt.ylim(None,None)
        plt.legend()


        ## Integrat energy for each timestepsol.chemical_potentials+
        absorbed_energy_dt = np.sum((sol.photon_densities[0]-sol.photon_densities[-1]).T*(self.par.E_i+self.par.E_f),1)
        absorbed_energy = np.array([np.trapz(absorbed_energy_dt[:i+1],sol.t[:i+1]) for i in range(len(absorbed_energy_dt))])
        factor = self.par.atomic_density * self.par.zstepsize # From energy per atom to energy per nm²
        total_free = np.sum(sol.E_free[:,:],0) * factor
        #total_free_simple = np.sum(sol.R_free[:,:]*(self.par.E_f),0) * factor
        total_inner = np.sum(sol.inner_energies[:,:],1) * factor
        total_inner = total_inner - total_inner[0]
        total_core = np.sum((self.par.M_core- sol.core[:,:])*self.par.E_f,0) * factor
        total_energies = total_free + total_inner + total_core
        #total_energies_simple = total_free_simple + total_inner + total_core

        plt.sca(axes[3, 1])
        #plt.figure()
        plt.plot(sol.t, total_core, label = 'Core')
        plt.plot(sol.t, total_inner, label = 'VB')
        plt.plot(sol.t, total_free, label = 'Free')
        #plt.plot(sol.t, total_free_simple, label = 'Free simple')
        plt.plot(sol.t, total_energies, label = 'Total in system')
        #plt.plot(sol.t, total_energies_simple, label = 'Total in system simple')
        plt.plot(sol.t, absorbed_energy, label = 'Total absorbed')

        ##Calculated in a different way just because
        #incident_pulse_energies_total = np.trapz((sol_photon_densities[0, :, :].T * (self.par.E_i + self.par.E_f)).T,
        #                                         x=sol.t)
        #incident_pulse_energies_total_check = self.par.I0_i * (self.par.E_i + self.par.E_f).T * self.par.atomic_density
        #transmitted_pulse_energies_total = np.trapz((sol_photon_densities[-1, :, :].T*(self.par.E_i+self.par.E_f)).T, x=sol.t)
        #absorbed_energy_total = incident_pulse_energies_total-transmitted_pulse_energies_total


        plt.axhline(absorbed_energy[-1], ls='--', label=f'Energy Lost: {100*(absorbed_energy[-1]-total_energies[-1])/absorbed_energy[-1]:.2f}%')
        plt.legend()

        plt.xlabel('Time (fs)')
        plt.ylabel('Energy (eV nm³ / atoms nm²)')

        plt.tight_layout()
        plt.show()
        #plt.pause(20)

        print('Done')
