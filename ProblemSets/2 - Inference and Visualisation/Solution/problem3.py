from __future__ import print_function

import numpy as np
import emcee
import _pickle as cPickle
import sys
import matplotlib.pyplot as plt
import corner
from utilities import pickle_to_file, pickle_from_file

def load_data():
    return pickle_from_file('../../../Datafiles/points_example1-py3.pkl')

    

def lnL(theta, x, y, yerr):
    """
    Minus the log likelihood of our simplistic Linear Regression (I return the 
    negative log likelihood since the function we use for optimiziation is a
    minimizer. 
    """
    a, b = theta
    model = b * x + a
    inv_sigma2 = 1.0/(yerr**2)
    
    return -0.5*(np.sum((y-model)**2*inv_sigma2))

def lnprior(theta):
    a, b = theta
    if -5.0 < a < 5.0 and -10.0 < b < 10.0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    """
    The likelihood to include in the MCMC.
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnL(theta, x, y, yerr)




def lnL_quad(theta, x, y, yerr):
    """
    The log likelihood in a quadratic regressor.
    """
    a, b, c = theta
    model = c*x*x + b * x + a
    inv_sigma2 = 1.0/(yerr**2)
    
    return -0.5*(np.sum((y-model)**2*inv_sigma2))

def lnprior_quad(theta):
    a, b, c = theta
    if -5.0 < a < 5.0 and -10.0 < b < 10.0 and -10.0 < c < 10.0:
        return 0.0
    return -np.inf

def lnprob_quad(theta, x, y, yerr):
    """
    The likelihood to include in the MCMC.
    """
    lp = lnprior_quad(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnL_quad(theta, x, y, yerr)

    
def run_emcee(x, y, y_obs, sigma, ml_result):

    # Set up the properties of the problem.
    ndim, nwalkers = 2, 100

    
    # Setup a bunch of starting positions.
    pos = [ml_result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
    # Create the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y_obs, sigma))

    sampler.run_mcmc(pos, 500)

    samples = sampler.chain[:, 50:, :].reshape((-1, 2))

    pickle_to_file((sampler.chain), 'emcee-linear.pkl')

    
    return sampler, samples




def run_emcee_quad(x, y, y_obs, sigma, ml_result):

    # Set up the properties of the problem.
    ndim, nwalkers = 3, 100

    # Run the linear ML version first to get a starting point - set the quadratic 
    # term to 0 as an initial guess.
    
    p_initial = np.append(result["x"], 0.1)
    p_initial = [0.0, 1.3, 0.08]
    
    # Setup a bunch of starting positions.
    pos = [p_initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
    # Create the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_quad, args=(x, y_obs, sigma))

    sampler.run_mcmc(pos, 500)

    samples = sampler.chain[:, 50:, :].reshape((-1, 3))
    
    return sampler, samples


def show_walkers(chain, labels, savefile=None):

    nwalkers, nreps, ndim = chain.shape

    xval = np.arange(0, nreps)

    fig, ax = plt.subplots(ncols=1, nrows=ndim, figsize=(14, 4))
    for i_dim in range(ndim):
        ax[i_dim].set_ylabel(labels[i_dim])
        
        for i in range(100):
            ax[i_dim].plot(xval, chain[i, :, i_dim], color='black', alpha=0.5)

    if savefile is not None:
        plt.savefig(savefile)

    return fig, ax


def show_corner_plot(samples,labels=['a', 'b'], savefile=None):

    fig = corner.corner(samples, labels=labels, truths=[0.0, 1.3],
                            quantiles=[0.16, 0.84])

    
    if savefile is not None:
        plt.savefig(savefile)

    return fig
