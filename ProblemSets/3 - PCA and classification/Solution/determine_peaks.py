#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from astropy.table import Table


def plot_mixture(X, m, xmin=-15, xmax=15):
    """
    A very simple plot of training data X and the Gaussian mixture m, showing the
    individual components with a dashed line.
    """

    # The x values we want to evaluate the model on.
    xplot = np.linspace(xmin, xmax, 1000)

    # Evaluate the mixture model at these x-values.
    logprob, p_component = M_best.score_samples(xplot[:, np.newaxis])

    # Convert to an actual probability. Fir the overall
    pdf = np.exp(logprob)

    # Then the individual components.
    pdf_individual = p_component * pdf[:, np.newaxis]

    plt.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
    plt.plot(xplot, pdf, '-k')
    plt.plot(xplot, pdf_individual, '--k')
    plt.text(0.04, 0.96, "Best-fit Mixture",
             ha='left', va='top', transform=ax.transAxes)
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.show()
    
t = Table().read('../../../Datafiles/mysterious-peaks.csv')
X = t['x'][:, np.newaxis]

# fit models with 1-10 components
N = range(1, 11)
models = [None for i in N]

for i, n in enumerate(N):
    models[i] = GaussianMixture(n).fit(X)

# compute the AIC and the BIC
AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]


print("The minimum number of components according to AIC is {0}".format(N[np.argmin(AIC)]))
print("The minimum number of components according to BIC is {0}".format(N[np.argmin(BIC)]))
    


fig = plt.figure(figsize=(12, 5))
fig.subplots_adjust(left=0.12, right=0.97,
                    bottom=0.21, top=0.9, wspace=0.5)


# plot 1: data + best-fit mixture according to AIC
ax = fig.add_subplot(131)
M_best = models[np.argmin(AIC)]

xplot = np.linspace(-15, 15, 1000)
logprob = M_best.score_samples(xplot[:, np.newaxis])
p_component = M_best.predict_proba(xplot[:, np.newaxis])
pdf = np.exp(logprob)
pdf_individual = p_component * pdf[:, np.newaxis]

ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
ax.plot(xplot, pdf, '-k')
ax.plot(xplot, pdf_individual, '--k')
ax.text(0.04, 0.96, "Best-fit Mixture according to AIC",
        ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')


# plot 1: data + best-fit mixture according to BIC
ax = fig.add_subplot(132)
Mb_best = models[np.argmin(BIC)]

xplot = np.linspace(-15, 15, 1000)
logprob = Mb_best.score_samples(xplot[:, np.newaxis])
p_component = Mb_best.predict_proba(xplot[:, np.newaxis])
pdf = np.exp(logprob)
pdf_individual = p_component * pdf[:, np.newaxis]

ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
ax.plot(xplot, pdf, '-k')
ax.plot(xplot, pdf_individual, '--k')
ax.text(0.04, 0.96, "Best-fit Mixture according to BIC",
        ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')


# plot 3: AIC and BIC
ax = fig.add_subplot(133)
ax.plot(N, AIC, '-k', label='AIC')
ax.plot(N, BIC, '--k', label='BIC')
ax.set_xlabel('N components')
ax.set_ylabel('information criterion')
ax.legend(loc=2)

plt.savefig('mysterious-peaks-results.pdf' ,format='pdf')

