import matplotlib.pyplot as plt
import numpy as np
import XNLdyn
import cvxpy
import scipy as sc
import scipy.interpolate

def run_modified_simulation(PAR, sim_options, changed_parameters, new_values, debug=False):
    print(f'Initializing a simulation where {changed_parameters} are changed to {new_values}\n')
    # Change parameters in PAR
    assert len(changed_parameters)==len(new_values)
    for i, par in enumerate(changed_parameters):
        setattr(PAR, par, new_values[i])
        
    PAR.make_derived_params()
    # Update the simulation and calculate derived parameters
    sim = XNLdyn.XNLsim(PAR)
    sim.DEBUG = debug
    # run it!
    incident, transmitted = sim.run(**sim_options)
    #print('Incident: ', incident)
    #print('Transmitted: ', transmitted)
    print('Transmission: ', 100 * transmitted/incident, ' %')
    return incident, transmitted

def make_integration_axis(Nsteps_r, sig):
    """
    Prepares the radial axis to integrate a radially symmetric spot.
    The number of steps <Nsteps_r> is distributed in irregular intervals to improve the sampling of a Gaussian spot.
    Especially relevant are the returns r_centers at which the simulation will run and the area dA for which that spot is valid.

    :param Nsteps_r: Number of points at which the simulation will run
    :param sig:  Sigma of the Gaussian
    :return: r_edges, r_centers, dr, dA
    """
    Nsteps_r +=1 # Evaluation points to edge-points
    Nsteps_r_inner = int(np.floor(Nsteps_r / 12))  # This many steps between 0 and sigma/4
    Nsteps_r_center = int(
        3 * np.floor(Nsteps_r / 4))  # This many between sigma/4 and 2.5*sigma (most relevant, highest derivative)
    Nsteps_r_outer = Nsteps_r - Nsteps_r_inner - Nsteps_r_center  # This many in the outer remaining region
    R = 5 * sig
    r_edges = np.concatenate((np.linspace(0, sig / 4, Nsteps_r_inner, endpoint=False),
                              np.linspace(sig / 4, 2.5 * sig, Nsteps_r_center, endpoint=False),
                              np.linspace(2.5 * sig, R, Nsteps_r_outer)))
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    dr = r_edges[1:] - r_edges[:-1]
    dA = r_centers * dr
    return r_centers, dr, dA

def spotgauss(r, sigma):
    """
    Returns a centered Gaussian distribution normalized to the area
    (in polar coordinates r and phi, but due to angular symmetry, phi is not required)
    :param r:    points in radius (vector)
    :param sigma: sigma of Gaussian
    :return: normalized amplitude
    """
    return 1/sigma**2 * np.exp(-(r**2)/(2*sigma**2))

def calculate_fluences(Nsteps_r, pulse_energy_max, pulse_profile_sigma):
    """
    :param Nsteps_r: Number
    :param pulse_energy_max: Joule
    :param pulse_profile_sigma: pulse_profile_sigma
    """
    r_centers, dr, dA = make_integration_axis(Nsteps_r, pulse_profile_sigma)
    
    fluences_phot_nm2 = pulse_energy_max * spotgauss(r_centers, pulse_profile_sigma)
    return fluences_phot_nm2, dA

def photons_per_J(photon_energy):
    echarge = 1.602176634e-19 #J/eV
    return 1/(photon_energy*echarge)


############## Convolution and Deconvolution
def gaussian_topnmorm(x, mu, sig):
    """ Gaussian, normalized to peak"""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lorentzian_topnorm(x,mu, gamma):
    """
    Loretzian peak function,
    centered around mu,
    half-with of gamma (FWHM = 2*gamma)
    Amplitude equals 1
    """
    denom = 1 + ((x-mu)/gamma)**2
    return 1 / denom

def pseudo_voigt_topnorm(x,mu,sig, eta):
    """
    sigma is both the sigma of the gaussian and the gamma of the lorentzian.
    eta is the share of the lorentzian, 1-eta the gaussian
    """
    return (eta*lorentzian_topnorm(x,mu,sig))+(1-eta)*gaussian_topnmorm(x,mu,sig)

def deconvolve_instrumentfcn(Enax, Spectrum, Working_Enax,instrument_sigma, flattening_factor = 800,\
                             plot = True, verbose = False, lorentzian_share = 0.5,\
                             max_iter = int(1e4)):
    """
    flattening_factor:  dampens oscillations. This is multiplied with the squared derivative of the deconvolution-solution in the cost function.
    
    """

    X = Working_Enax
    dE = np.mean(X[1:]-X[:-1])

    Y = sc.interpolate.interp1d(Enax,Spectrum, kind = 'cubic', bounds_error=False, fill_value=np.nan)(X)  # Y-values (usually the spectrum)
    good = np.isfinite(Y)
    Y = Y[good]
    X = X[good]
    
    Nsig = len(X)

    
    # Determine kernels
    X_kernel_half = np.arange(0,10*instrument_sigma,dE)
    X_kernel = np.concatenate((-X_kernel_half[::-1],X_kernel_half[1:])) # this makes sure it contains a point at 0, which is  important to make a good delta function
    Nkern = len(X_kernel)
    c = pseudo_voigt_topnorm(X_kernel,0,instrument_sigma, lorentzian_share)
    c=c/np.sum(c)
    c_delta = gaussian_topnmorm(X_kernel,0,1e-6) # Delta Kernel to make an unmodified convolved spectrum

    Nconv = Nsig+Nkern-1 # Points in convolution

    Yconv = np.convolve(c_delta,Y) #Unmodified convolved spectrum
    
    
    
    #Perform deconvolution
    deconvolution = cvx.Variable(Nsig)
    objective = cvx.sum_squares(cvx.conv(c, deconvolution)[:,0] - Yconv) + flattening_factor*cvx.sum(cvx.diff(deconvolution)**2)
    prob = cvx.Problem(cvx.Minimize(objective), [deconvolution >= 0])
    prob.solve(solver=cvx.SCS, verbose=verbose, eps = 5e-5, max_iters=max_iter)
        
    if prob.solver_stats.num_iters == 1e4:
        print('Warning - maximum iteration reached!')

    def normmax(V):
        return V/np.nanmax(V)
    
    if plot is True:
        fig, axes = plt.subplots(2,1)
        plt.sca(axes[0])
        plt.plot(Yconv, label = 'Input convolved with delta')
        plt.plot(np.convolve(deconvolution.value, c), label = 'Re-convolution')
        plt.legend()

        plt.sca(axes[1])
        plt.plot(X,normmax(Y), label = 'input')
        plt.plot(X_kernel+np.mean(X),normmax(c), label ='kernel (centered)')
        plt.plot(X_kernel+np.mean(X),normmax(c_delta), label ='delta - kernel (centered)')
        plt.plot(X,normmax(deconvolution.value), label = 'deconvolved solution')
        #plt.plot(X,normmax(np.convolve(deconvolution.value, c,mode='same')/np.sum(c)),'.', label = 'reconvolved')

        plt.legend()
    return deconvolution.value, X#*np.max(Y)/np.max(deconvolution.value)
    
def reconvolve(E, Spec, instrument_sigma, lorentzian_share = 0.5):
    dE = np.mean(E[1:]-E[:-1])
    X_kernel_half = np.arange(0,10*instrument_sigma,dE)
    X_kernel = np.concatenate((-X_kernel_half[::-1],X_kernel_half[1:])) # this makes sure it contains a point at 0, which is  important to make a good delta function
    
    c = pseudo_voigt_topnorm(X_kernel,0,instrument_sigma, lorentzian_share)

    c/=np.sum(c)
    Yconv = np.convolve(Spec,c/np.sum(c), mode='same') #Unmodified convolved spectrum
    return Yconv
