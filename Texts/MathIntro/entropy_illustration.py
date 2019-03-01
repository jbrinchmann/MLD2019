#!/usr/bin/env python

import JBMath.PDF as PDF
import JBMath.GaussianPDF as gPDF
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4)

sigmas = np.array([0.1, 0.5, 1.5, 10])


for i, s in enumerate(sigmas):
    g = gPDF.GaussianPDF(0.0, s)

    ent = g.entropy()
    x = g.x
    y = g.pdf

    axes[i].plot(x, y)
    axes[i].set_xlim(-10, 10)
    axes[i].set_title("Entropy = {0:.3f}".format(ent))
    
    axes[i].text(-9, 0.95*np.max(y), r'$\sigma={0:.1f}$'.format(s), color='red')
    axes[i].get_yaxis().set_ticks([])
    
fig.set_size_inches(15, 4)
plt.savefig("entropy_example.png")

