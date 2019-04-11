
# coding: utf-8

# # Do a PCA decomposition of the Pickles library
# 
# This can be done in a variety of ways - depending on what data we focus on.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

# ## Loading of the library of spectra
# 
# The spectra are stored in a FITS file in the Pickles subdirectory called `pickles-spectra.fits`. This contains the wavelength axis in the first HDU, the flux in the next and the flux uncertainty in the last. But we also need to get the overview table which has the classification of the spectra.

def load_pickles_library(fname='pickles-spectra.fits'):
    hdul = fits.open(fname)
    wave = hdul[0].data
    flux = hdul[1].data
    dflux = hdul[2].data
    
    return wave, flux, dflux

def load_overview_table(fname='overview-of-spectra.vot'):
    return Table().read(fname)


def run_PCA(wavelength_range=[3000, 10000], n_components=10):
    """Driver routine for running PCA"""

    wave, flux, dflux = load_pickles_library()
    t_overview = load_overview_table()

    i_use, = np.where((wave > wavelength_range[0]) &
                          (wave < wavelength_range[1]))

    flux_use = flux[i_use, :]
    dflux_use = dflux[i_use, :]
    wave_use = wave[i_use]

    # This creates the design matrix - note that I need
    # to transpose to match the convention used by sklearn
    X = flux_use.T.copy()
    
    # Some illustrations
    show_example_spectra(t_overview, wave_use, flux_use, dflux_use,
                             wavelength_range=wavelength_range,
                             outfile='example-spectra-Pickles.pdf')
    

    # This does automatic whitening
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X)

    show_explained_variance(pca, outfile='explained-variance-Pickles.pdf')
    show_eigenspectra(pca, wave_use/1e4, outfile='example-eigenspectra-Pickles.pdf')
    
    pcs = pca.transform(X)
#    show_pca_pairplots(pcs, t_overview)
    
    sp_depr = pca.inverse_transform(pcs)
    illustrate_reconstruction_of_spectra(sp_depr, wave_use/1e4, flux_use, dflux_use, t_overview)
    
    

def show_pca_pairplots(pcs, t_overview):
    # Visualise the PCs
    #
    # For this we use the pcs that is calculated above. Here it is interesting
    # to show this as a pair plot coloured by MKType and by luminosity class. 
    
    # For this it is convenient to convert to a Pandas data frame.
    dft = pd.DataFrame(pcs[:, 0:4], columns=('PC1', 'PC2', 'PC3', 'PC4'))
    
    dft['type'] = t_overview['SPType']
    dft['lumclass'] = t_overview['Lumclass']
    
    sns.pairplot(dft, hue='type', vars=('PC1', 'PC2', 'PC3', 'PC4'),
                plot_kws={'s': 75, 'alpha': 0.5})
    plt.savefig('PC-pairplot-MKclass.pdf')

    sns.pairplot(dft, hue='lumclass', vars=('PC1', 'PC2', 'PC3', 'PC4'))
    plt.savefig('PC-pairplot-lumclass.pdf')

    

    # so it seems as if the luminosity class does not really correlate with anything very well while there is a correlation between MK class and various PCs. We can explore this further by using a 1D plot:


    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    
    todo = ['PC1', 'PC2', 'PC3']
    for i, t_pc in enumerate(todo):
        ax[i].scatter(dft[t_pc], t_overview['numtype'])
        ax[i].set_xlabel(t_pc)
    
    ax[0].set_ylabel('MK Type')
    plt.savefig('PCs-vs-MKclass.pdf')
    


    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

    todo = ['PC1', 'PC2', 'PC3']
    for i, t_pc in enumerate(todo):
        ax[i].scatter(dft[t_pc], t_overview['numlclass'])
        ax[i].set_xlabel(t_pc)
    
    ax[0].set_ylabel('Luminosity class')
    fig.savefig('PCs-vs-lumclass.pdf')

    
def show_eigenspectra(pca, wave, outfile=None):
    # ## Looking at eigenspectra
    # 
    # Let us now see what the first few eigenspectra look like

    
    fig, ax = plt.subplots(nrows=3, figsize=(15, 15))
    for i in range(3):
        ax[i].plot(wave, pca.components_[i, :])
        ax[i].set_ylabel('PC{0}'.format(i))

    ax[2].set_xlabel('Wavelength [µm]')
    if outfile is not None:
        fig.savefig(outfile)



def show_explained_variance(pca, outfile=None):
    """Plot the explained variance plot"""

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    ax.set_ylim(0, 0.05)
    ax.set_xlabel('Eigenvalues')
    ax.set_ylabel('Explained variance')
    if outfile is not None:
        fig.savefig(outfile)

    return fig, ax

def show_example_spectra(t_overview, wave, flux, dflux,
                             wavelength_range=None, outfile=None):
    """Plot the first spectrum in each class."""

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,7))
    if (wavelength_range is None):
        xplot = np.log10(wave)
        xlabel = r'Log wavelength [\AA]'
    else:
        xplot = wave/1e4
        xlabel = 'Wavelength [µm]'
        
    MKclasses = ['o', 'b', 'a', 'f', 'g', 'k', 'm']
    for MK in MKclasses:
        ii, = np.where(t_overview['SPType'] == MK)

        lbl = 'Class={0} [{1} spectra]'.format(MK, len(ii))

        print("Doing "+lbl)
        ax.plot(xplot, flux[:, ii[0]], label=lbl)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$F(\lambda)$')
    ax.legend()
#    if wavelength_range is not None:
#        ax.set_xlim(wavelength_range)

    if outfile is not None:
        fig.savefig(outfile)

        
    return fig, ax


def show_mean_spectrum(wave, flux, outfile=None):
    
    # This is the design matrix (note the transpose to adhere to the sklearn convention)
    X = flux.T.copy()
    
    # And we want to calculate a mean spectrum too.
    mean_spectrum = np.sum(X, axis=0)/len(X[:,0])
    
    fig, ax = plt.subplots(ncols=1, figsize=(12,5))
    ax.plot(wave_opt, mean_spectrum)
    ax.set_xlabel('Wavelength [µm]')
    ax.set_ylabel(r'$F(\lambda)$')
    ax.set_title('Mean spectrum')
    if (outfile is not None):
        fig.savefig(outfile)
                   
    



def illustrate_reconstruction_of_spectra(sp_depr, wave, flux, dflux, t_overview):

    # # Reconstruction of spectra
    # 
    # We now want to reconstruct spectra and see how well that works.
    

    

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wave, flux[:, 30], color='black', label='original')
    ax.plot(wave, sp_depr[30, :], color='red', label='recon. 10 PC')
    ax.legend()
    ax.set_xlabel('Wavelength [µm]')
    ax.set_ylabel('Flux')
    fig.savefig('example-PCA-reconstruction.pdf')



    MKclasses = ['o', 'b', 'a', 'f', 'g', 'k', 'm']
    MKclasses = ['b', 'a', 'k', 'm']
    fig, ax = plt.subplots(figsize=(12, 12), nrows=len(MKclasses))
    for i, MK in enumerate(MKclasses):
        ii, = np.where(t_overview['SPType'] == MK)
        ii = ii[0]
        diff = (flux[:, ii]-sp_depr[ii, :])
    
        # Not every spectrum has an error estimate at all wavelengths.
        # To get a meaningful (ie. comparable) weighted RSS I want to normalise
        # by this.
        i_with_err, = np.where(dflux[:, ii] > 0)
        n_with_err = len(i_with_err)
        rss = np.sum(diff**2)
        rss_w = np.nansum((diff[i_with_err]/dflux[i_with_err, ii])**2)/n_with_err
    
        ax[i].plot(wave, diff, color='black')
        ax[i].fill_between(wave, dflux[:, ii], -dflux[:, ii],
                        color='#00aaff', alpha=0.5)
        ax[i].text(0.9, 0.9, ' RSS={0:.3f}'.format(rss), transform=ax[i].transAxes)
        ax[i].text(0.9, 0.8, r'$\chi^2={0:.2f}$'.format(rss_w), transform=ax[i].transAxes)
        ax[i].text(0.9, 0.7, 'file={0}'.format(t_overview['file'][ii][2:]), transform=ax[i].transAxes)

    fig.savefig('example-PCA-reconstruction-residual.pdf')

# # Determining the number of components needed
# 
# This requires some more calculation. Let us approach this from a cross-validation point of view. In that case we need to set up a cross-validation run, loop over this and run PCA with varying numbers of components and calculate the RSS over the whole sample.

# # In[27]:


# from sklearn.model_selection import KFold

# X = flux.T
# def do_one_cv_pca(X, n_components, n_splits=10):
#     # Create folds for N data points in n_folds:
        
#     kf = KFold(n_splits=n_splits)
#     RSS = 0.0
#     for train, test in kf.split(X):
#         x_train = X[train, :]
#         x_test = X[test, :]
    
#         # Calculate PCA on the training sample.
#         # This does automatic whitening
#         pca = PCA(n_components=n_components, whiten=True)
#         pca.fit(x_train)
    
#         # Use this to reconstruct the test sample
#         #pcs = pca.transform(x_test)
#         #sp_depr = pca.inverse_transform(pcs)
    
#         #RSS_this = np.sum((x_test-sp_depr)**2)
#         RSS_this = pca.score(x_test)
#         RSS += RSS_this
    
#     return RSS


# def do_multiple_n_comp(X):
    
#     n_comps = np.arange(1, 50)
# #    n_comps = np.arange(1, 120, 5)
#     N = len(n_comps)
#     RSS = np.zeros(N)
#     for i, n_comp in enumerate(n_comps):
#         print("Doing {0:2d}".format(i))
#         RSS[i] = do_one_cv_pca(X, n_comp)
        
#     return RSS, n_comps


# # In[28]:


# RSS = do_one_cv_pca(X, 5)


# # In[29]:


# # RSSes, n_comps = do_multiple_n_comp(X)


# # In[30]:


# # plt.plot(n_comps, np.exp(RSSes/1e7))


# # In[32]:


# t_overview



# run_PCA()
