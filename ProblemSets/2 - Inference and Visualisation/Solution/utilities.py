"""
Routines to for problemset 3. 

"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import cPickle
except:
    import _pickle as cPickle
import sys

sns.set()

def pickle_to_file(data, fname):

    try:
        fh = open(fname, 'wb')
        cPickle.dump(data, fh)
        fh.close()
    except:
        print("Pickling failed!", sys.exc_info()[0])

def pickle_from_file(fname):

    try:
        fh = open(fname, 'rb')
        data = cPickle.load(fh)
        fh.close()
    except:
        print("Loading pickled data failed!", sys.exc_info()[0])
        data = None

    return data


def datafile():
    return 'linear-plus-freq.pkl'

def create_linear_plus_frequency():
    """
    Create the input dataset. This is a constant value with a sinusoidal offset.

    y = lvl + N(0, sigma^2) + A*sin(t*k)

    t is given in seconds, so k has units 1/s - or Hz. 
    
    """

    lvl = 3.0
    sigma = 0.05 # Constant error.

    k = 0.4     # Hz
    A = 0.3

    t = np.linspace(0, 60, 120)
    n_t = len(t)

    flux = lvl + np.random.randn(n_t)*sigma + A*np.sin(t*k)

    fname = datafile()
    d = {'flux': flux, 'A': A, 'k': k, 'level': lvl, 'sigma': sigma,
         't': t, 'n_t': n_t, 'dy': np.zeros(n_t)+sigma}
    pickle_to_file(d, fname)
    
    return d


def show_full_dataset():

    d = pickle_from_file(datafile())

    plt.plot(d['t'], d['flux'], 'o')
    plt.show()


def extract_subset(d, N, seed=15):
    inds = np.arange(d['n_t'])

    x = d['t']
    y = d['flux']
    
    np.random.seed(seed)
    np.random.shuffle(inds)

    x = x[inds[0:N]]
    y = y[inds[0:N]]
    dy = d['dy'][0:N]

    return x, y, dy
    
def show_subset_of_dataset(N, seed=0, include_true=False):

    d = pickle_from_file(datafile())

    x = d['t']
    y = d['flux']
    inds = np.arange(d['n_t'])

    np.random.seed(seed)
    np.random.shuffle(inds)

    x = x[inds[0:N]]
    y = y[inds[0:N]]
    dy = d['dy'][0:N]

    fig, ax = plt.subplots()
    ax.errorbar(x, y, dy, fmt='ok', ecolor='gray')
    ax.set(xlabel='time', ylabel='flux', title='subset')

    if include_true:
        y_true = d['level']+ d['A']*np.sin(d['t']*d['k'])
        ax.plot(d['t'], y_true, 'r')
    
    fig.show()

#--------------------------------    
# Likelihood function
#--------------------------------    

def lnL(theta, x, y, yerr):
    """
    The log of our model for the system.

    y = lvl + A*sin(t*k)
    """
    lvl, A, k, = theta
    model = lvl + A*np.sin(x*k)
     
    inv_sigma2 = 1.0/yerr**2

    return -0.5*(np.sum((y-model)**2*inv_sigma2))


# Priors
def lnL_prior(theta):
    """
    This is a fairly broad prior
    """
    lvl, A, k, = theta
    if 0 < lvl < 10 and 0 < A < 3 and 0 < k < 2:
        return 0.0
    else:
        return -np.inf


def lnProb(theta, x, y, yerr):
    """
    The full likelihood
    """
    lp = lnL_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp+lnL(theta, x, y, yerr)


    
    
