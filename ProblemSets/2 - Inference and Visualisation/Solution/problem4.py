from __future__ import print_function
import numpy as np
from scipy.stats import norm
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from astroML.decorators import pickle_results
import os.path
import utilities as u
import emcee
import corner

def calc_polynomial(theta, x):
    """
    Calculate a polynomial function.
    """
    
    return np.polyval(theta[::-1], x)

def lnL(theta, data, model=calc_polynomial):
    """
    Calculate the log likelihood of the fit.
    """

    if data is None:
        raise
    
    # Get the data
    x, y, sigma = data

    # Calculate the model for these parameters
    y_fit = model(theta, x)

    inv_sigma2 = 1.0/(sigma**2)
    ln_pdf = -0.5*(np.sum(np.log(2*np.pi*sigma**2)+ (y-y_fit)**2*inv_sigma2))

    return ln_pdf

def neg_lnL(theta, data, model=calc_polynomial, *args):
    return -lnL(theta, data, model=model)
        
def lnL_alt1(theta, model=calc_polynomial, data=None):
    """
    Calculate the log likelihood of the fit.
    """

    # Get the data
    x, y, sigma = data

    # Calculate the model for these parameters
    y_fit = model(theta, x)

    # Alternative version 1:
    ln_pdf = 0.0
    for y_v, y_fit_v, sigma_v in zip(y, y_fit, sigma):
        ln_pdf += norm.logpdf(y_v, y_fit_v, sigma_v)

    return ln_pdf

def lnL_alt2(theta, model=calc_polynomial, data=None):
    """
    Calculate the log likelihood of the fit.
    """

    # Get the data
    x, y, sigma = data

    # Calculate the model for these parameters
    y_fit = model(theta, x)

    # Alternative version 2:
    ln_pdf = sum(norm.logpdf(*args)
                 for args in zip(y, y_fit, sigma))

    return ln_pdf


def my_polyfit(data, degree, model=calc_polynomial):
    """
    Carry out a polynomial fit to the data. We could also achieve
    this using the polyfit function in numpy with weighting. This uses
    the likelihood explicitly.

    """

    # An array with initial guesses for the parameters.
    theta_init = (degree + 1) * [0]
    neg_logL = lambda theta: -lnL(theta, data, model=model)
    res = optimize.minimize(neg_logL, theta_init)

    return res['x']
    
def comparison_fits(data, outfile='poly-comparison.pkl', load=True):
    """
    Compare fits of different polynomial power to the data. 
    """

    if load and os.path.isfile(outfile):
        print("Loading from {0}".format(outfile))
        return u.pickle_from_file(outfile)
    
    degrees = np.arange(1, 10)
    model = calc_polynomial

    # Find the best-fit polynomial for the different degrees. 
    thetas = [my_polyfit(data, d) for d in degrees]

    # Calculate the maximum likelihood at the best-fit values of 
    # theta
    logL_max = [lnL(theta, data, model=model) for theta in thetas]

    return {'thetas': thetas, 'logL_max': logL_max, 'degrees': degrees}



    
def plot_fit_comparison(data, res, outfile=None):
    """
Create a plot of the previous run - illustrating the fit of
a degree 1, 2 and 9 polynomial to the data.
    """
    x, y, sigma_y = data
    xfit = np.linspace(0, np.max(x)*1.05, 1000)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(res['degrees'], res['logL_max'])
    ax[0].set(xlabel='degree', ylabel='log(Lmax)')
    ax[1].errorbar(x, y, sigma_y, fmt='ok', ecolor='gray')
    ylim = ax[1].get_ylim()
    for (degree, theta) in zip(res['degrees'], res['thetas']):
        if degree not in [1, 2, 9]: continue
        ax[1].plot(xfit, calc_polynomial(theta, xfit),
                   label='degree={0}'.format(degree))
        ax[1].set(ylim=ylim, xlabel='x', ylabel='y')
        ax[1].legend(fontsize=14, loc='best');

    if outfile is not None:
        plt.savefig(outfile, format='pdf')
    else:
        fig.show()


def AIC(lnL_max, k):
    return 2*k - 2*lnL_max

def BIC(lnL_max, n, k):
    return - 2*lnL_max + k*np.log(n)




def plot_BIC_AIC(data, res, outfile=None):
    """
    Plot the Bayesian and Akaike information criterion against
    the polynomial power
    """
    x, y, sigma_y = data
    # The number of free parameters to calculate
    N = len(x)
    BIC_values = [BIC(lnL_max, N, len(theta)) for lnL_max, theta in
                  zip(res['logL_max'], res['thetas'])]

    AIC_values = [AIC(lnL_max, len(theta)) for lnL_max, theta in
                  zip(res['logL_max'], res['thetas'])]

    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(res['degrees'], BIC_values, color='red', label='BIC')
    ax.plot(res['degrees'], AIC_values, color='blue', label='AIC')
    ax.set(xlabel='degree', ylabel='IC')
    ax.legend(fontsize=14, loc='best')

    if outfile is not None:
        plt.savefig(outfile, format='pdf')
    else:
        fig.show()


def ln_prior(theta):
    # This is a prior constant between +/- 100 - the normalisation
    # constant is then 1/200 in each direction, so n_theta/200.
    #
    # Note that you need to normalise the prior properly here. 
    #
    if np.any(abs(theta) > 100):
        return -np.inf  # log(0)
    else:
        return np.log(200**(-len(theta)))
    
def ln_posterior(theta, data):
    theta = np.asarray(theta)

    return ln_prior(theta) + lnL(theta, data)


def run_MCMC(degree, data,
             ln_posterior=ln_posterior,
             nwalkers=50, nburn=1000, nsteps=2000):
    """
Run emcee for polynomial fit with degree degree.
    """
    
    ndim = degree + 1  # this determines the model
    rng = np.random.RandomState(0)

    # Get a M-L starting guess.
    # res = my_polyfit(data, degree)
    #starting_guesses = [res + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    starting_guesses = rng.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                    args=[data])
    sampler.run_mcmc(starting_guesses, nsteps)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    lnprob = sampler.lnprobability[:,nburn:].reshape(-1, 1)
    return trace, lnprob, sampler


def show_corner_plot(samples, labels=None, savefile=None):

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.84])

    
    if savefile is not None:
        plt.savefig(savefile)

    return fig

def integrate_posterior_2D(log_posterior, xlim, ylim, data):
    func = lambda theta1, theta0: np.exp(log_posterior([theta0, theta1], data))
    return integrate.dblquad(func, xlim[0], xlim[1],
                             lambda x: ylim[0], lambda x: ylim[1])

def integrate_posterior_3D(log_posterior, xlim, ylim, zlim, data):
    func = lambda theta2, theta1, theta0: np.exp(log_posterior([theta0, theta1, theta2], data))
    return integrate.tplquad(func, xlim[0], xlim[1],
                             lambda x: ylim[0], lambda x: ylim[1],
                             lambda x, y: zlim[0], lambda x, y: zlim[1])
    
