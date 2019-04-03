#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from astroML.decorators import pickle_results
from astroML.linear_model import LinearRegression

import scipy.optimize as op


def func_line(x, a, b):
    return a + b*x

def func_pow2(x, a, b, c):
    return a + b*x + c*x*x

@pickle_results('points_example1.pkl')
def generate_points():

    #
    # This is the model generating the data.
    #
    x = np.linspace(-3, 3, 10)
    y = 1.3*x + 0.08*x*x
    n_y = len(y)

    np.random.seed(100)
    sigma = np.sqrt(np.abs(y))/10+0.4
    dy = np.random.normal(0, 1, size=n_y)*sigma

    y_obs = y+dy

    return (x, y, y_obs, sigma)


def neglnL(theta, x, y, yerr):
    """
    Minus the log likelihood of our simplistic Linear Regression (I return the 
    negative log likelihood since the function we use for optimiziation is a
    minimizer. 
    """
    a, b = theta
    model = b * x + a
    inv_sigma2 = 1.0/(yerr**2)
    
    return 0.5*(np.sum((y-model)**2*inv_sigma2))



def run_ml():

    x, y, y_obs, sigma = generate_points()

    result = op.minimize(neglnL, [1.0, 0.0], args=(x, y_obs, sigma))

    a_ml, b_ml = result["x"]

    return result

    

def run_LinearRegression_fit():
    """
    Run the fit but now using astroML's LinearRegression
    """
    x, y, y_obs, sigma = generate_points()
    m = LinearRegression()
    m.fit(x[:, None], y_obs, sigma)
    print("Intercept={0} Slope={1}".format(m.coef_[0], m.coef_[1]))
    

def run_curvefit(x, y_obs, sigma, y):
    """
    A simple fit of a linear model to the data using curve_fit.
    """

    pars, cov = curve_fit(func_line, x, y_obs)
    y_pred = func_line(x, pars[0], pars[1])
    chi2_1 = np.sum((y_pred-y_obs)**2/sigma**2)

    pars2, cov2 = curve_fit(func_pow2, x, y_obs)
    y_pred2 = func_pow2(x, pars2[0], pars2[1], pars2[2])
    chi2_2 = np.sum((y_pred2-y_obs)**2/sigma**2)


    print("intercept = {0}  slope={1}".format(pars[0], pars[1]))
    print("Chi2(linear)={0}".format(chi2_1))
    print("Chi2(quad)={0}".format(chi2_2))
    
    
    plt.scatter(x, y_obs)
    plt.errorbar(x, y_obs, yerr=sigma, fmt='o')
    plt.plot(x, y, 'g--')

    plt.plot(x, y_pred, 'r')
    #plt.plot(x, y_pred2, 'b--')

    plt.text(-3.5, 5, 'True', color='g')
    plt.text(-3.5, 4.5, 'Linear fit', color='r')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('example-ml-curvefit.pdf', format='pdf')
    plt.show()


