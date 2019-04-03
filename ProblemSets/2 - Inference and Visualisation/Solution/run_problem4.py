#!/usr/bin/env python

import problem4 as p4
import numpy as np
import utilities as u
import os.path
import corner

def discrete_integrate(trace, lnprob):
    """
    Use a multi-D histogram to do the integral. This is
    considerably harder to do when doing
    high-dimensional data.
    """ 
    
    (n_trace, n_dim) = trace.shape

    h, bins = np.histogramdd(trace, weights=np.exp(lnprob))

    deltas = [x[1]-x[0] for x in bins]

    val = np.sum(h)
    for dx in deltas:
        val *= dx

    return val

# Get the data
data = u.pickle_from_file('data-for-poly-test.pkl')
x, y, sigma_y = (data['x'], data['y'], data['sigma_y'])
data = (x, y, sigma_y)
    
# First the polynomial comparisons
print "Doing polynomial evaluation"
res = p4.comparison_fits(data, outfile='poly-comparison.pkl')
p4.plot_fit_comparison(data, res, outfile='fit-comparison.pdf')
p4.plot_BIC_AIC(data, res, outfile='fit-AIC-BIC.pdf')


deg1_file = 'mcmc-poly1.pkl'
if os.path.isfile(deg1_file):
    tr1, lnp1 = u.pickle_from_file(deg1_file)
else:
    print "Running the MCMC calculation for a degree n=1 poly."
    
    tr1, lnp1 = p4.run_MCMC(1, data)
    u.pickle_to_file((tr1, lnp1), deg1_file)


deg2_file = 'mcmc-poly2.pkl'
if os.path.isfile(deg2_file):
    tr2, lnp2 = u.pickle_from_file(deg2_file)
else:
    print "Running the MCMC calculation for a degree n=2 poly."
    tr2, lnp2 = p4.run_MCMC(2, data)
    u.pickle_to_file((tr2, lnp2), deg2_file)

I1 = discrete_integrate(tr1, lnp1[:, 0])
I2 = discrete_integrate(tr2, lnp2[:, 0])
bayes_factor = I1/I2

print "Discrete integral #1: {0}".format(I1)
print "Discrete integral #2: {0}".format(I2)
print "   Bayes factor O_12: {0}".format(bayes_factor)

fig = corner.corner(tr1, labels=['a', 'b'])
fig.savefig('corner-linear.pdf')

fig = corner.corner(tr2, truths=[-0.2, 1, 0.4], labels=['a', 'b', 'c'])
fig.savefig('corner-quad.pdf')

# Now direct integration:
#xlim, ylim = zip(tr1.min(0), tr1.max(0))
#Z1, err_Z1 = p4.integrate_posterior_2D(p4.ln_posterior, xlim, ylim, data)
#print("Z1 =", Z1, "+/-", err_Z1)


#xlim, ylim, zlim = zip(tr2.min(0), tr2.max(0))
#Z2, err_Z2 = p4.integrate_posterior_3D(p4.ln_posterior, xlim, ylim, zlim, data)
#print("Z2 =", Z2, "+/-", err_Z2)

#print "O_12 from direct integration={0}".format(Z1/Z2)

