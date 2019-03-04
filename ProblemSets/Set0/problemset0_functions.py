import pyfits
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator

# Some of the data in our tables are invalid - but we know this 
# and deal with it later - this command will silence the warnings that 
# are otherwise printed out.
np.seterr(invalid='ignore')

def read_exoplanet_table():
    """
    Read in the exoplanet table from exoplanet.eu in FITS format.
    """

    # The name of the file
    file = '../Catalogues/exoplanet_catalog-2016-10-12.fits'

    # Read in the FITS table
    t = pyfits.getdata(file, 1)

    # Return the result to the user.
    return t


def count_Kepler(t):
    """
    Count the number of planets detected by Kepler assuming that if the name
    starts with Kepler it was detected by that satellite. 
    """

    counter = 0
    for n in t['name']:
        if (n.startswith('Kepler')):
            counter = counter+1

    return counter



def task1():
    """
    This function carries out task 1 in the problem set - namely to count the number of
    exoplanets detected by Kepler. This is defined on page 8 of the problem set file
    """

    t = read_exoplanet_table()
    n_Kepler = count_Kepler(t)

    print "--------------------------------------------------------"
    print "Task 1: How many of the planets were found by Kepler\n"
    print "The Kepler mission has detected {0} planets in the file I have available.".format(n_Kepler)
    print "\n\n"
    

def meanMass_function(t):
    """
    Trivial function to calculate the mean mass of planets in table t.
    """

    sumMass = 0.0
    count = 0
    for m in t['mass']:
        if np.isfinite(m):
            sumMass += m
            count += 1

    meanMass = sumMass/count
    return(meanMass)

    
    
def task2():
    """
    This function carries out task 2 in the problem set.
    This is defined on page 8 of the problem set file.
    """

    t = read_exoplanet_table()

    #
    # Now, there is a challenge here - because some data are not available. For those
    # planets that do not have mass estimates, the table contains NaN (not a Number). It is
    # best to select out those that have valid mass estimates before continuing.
    #
    useful, = np.where(np.isfinite(t['mass']))
    n_all = len(t['mass'])
    n_good = len(useful)
    n_bad = n_all - n_good

    # This variable will contain the masses of the planets for we know the mass.
    masses = t.mass[useful]

    # We can now use np.mean to get the mean mass. If I had not filtered out the missing 
    # mass estimates, I would use np.nanmean instead.
    meanMass = np.mean(masses)

    # I also get the mean mass using the user-written function
    meanMass2 = meanMass_function(t)

    # It is also useful to compare this arithmetic mean to the geometric mean = exp( mean(ln(x)))
    geometricMean = np.exp(np.mean(np.log(masses)))

    # Output the results
    print "--------------------------------------------------------"
    print "Task 2: The mean mass of planets\n"
    print "A total of {0} planets ({1:.2f}%) have unknown mass - I ignore these\n".format(n_bad, 100.0*n_bad/n_all)
    print "The remaining planets have a mean mass = {0:.2f} MJup and a geometric mean mass of {1:.2f} MJup".format(meanMass, geometricMean)
    print "Finally, the difference between my function and numpy.mean's calculation = {0:.3f}\n".format(meanMass-meanMass2)


def task3():
    """
This function does task 3 a, b, c - the plotting of an HR diagram. 
    """
    
    t = read_exoplanet_table()

    # The output PDF files
    file_step1 = 'simple-HR-diagram.pdf'
    file_step2 = 'HR-diagram-Mcoloured.pdf'
    file_step3 = 'HR-diagram-Mcoloured-scaled.pdf'

    print "The HR diagram without scaling or colour."
    print "Written to {0}".format(file_step1)
    showHR(t, pdf=file_step1, includegrid=True)

    print "The HR diagram - points coloured by log stellar mass"
    print "Written to {0}".format(file_step2)
    showHR(t, colour='log mass', pdf=file_step2)
    
    print "The HR diagram - points coloured by log stellar mass amd scaled by stellar radius"
    print "Written to {0}".format(file_step3)
    showHR(t, colour='log mass', scale='radius', pdf=file_step3)

    
    

def showHR(t, scale=None, colour=None, pdf=None, includegrid=False):
    """
     Show a Hertzsprung-Russel diagram from the data in the 
     table dictionary, t

     t - a table dictionary, as e.g. returned from reading a FITS table. 
     scale - Quantity to use to scale symbols. 'radius' scales
             by star_radius (default=None)
     colour - Quantity to use to colour symbols. 'log radius'
              scales by the log10 of star_radius constrained to
              lie between 0.1 and 20. 'log mass' scales by the
              log10 of star_mass - the logi is scaled to lie 
              between -1 and 0.5 (default=None).
     pdf - if set to a string, save a PDF to a file with name
           taken from the string. No checks are made.
           (default = None)
     includegrid - if set to True, plot a grid on top of the
                   plot grid (default=False).
     
    """
    
    # First get an absolute magnitude
    Vabs = t['mag_v'] - 5*np.log10(t['star_distance']/10.0)
    Teff = t['star_teff']

    # The user can also ask for scaling according to the mass of the planet. 
    if scale == 'radius':
        # Scale this between 0.5 and 20
        scaling = np.clip(t['star_radius'], 0.5, 15)*2
    else:
        scaling = np.ones(len(Teff))*4


    # The user might also request a colour scaling - we handle this here. 
    if colour == 'log radius':
        # Scale this between 0.1 and 20
        tmp_scaling = np.clip(t['star_radius'], 0.1, 20)
        col = np.log10(tmp_scaling)
        col = col/np.nanmax(col)
    elif colour == 'log mass':
        # The clip region was chosen after looking at the histogram in topcat. 
        col = np.clip(np.log10(t['star_mass']), -1, 0.5)
        # I then transform the values so they lie between 0 and 1.
        col = col - np.nanmin(col)
        col = col/np.nanmax(col)
    else:
        col = 'r'

    
    # Not all points are acceptable - select only the good ones but
    # report what fraction was bad.
    good,  = np.where((Teff > 1500) & (Vabs < 30) & (Vabs > -5)
                      & (scaling > 0) & (t['star_radius'] > 0.01))
    n_good = len(good)
    n_all = len(Teff)
    n_bad = n_all - n_good
    frac_bad = n_bad/np.float(n_all)

    print "I will plot {0:d} points, that is {1:.1f}% of the total".format(n_good, 100*frac_bad)
    Teff = Teff[good]
    Vabs = Vabs[good]
    scaling = scaling[good]
    try:
        col = col[good]
    except:
        pass

    #    print "Colors=", col
    
    # Set up the plot area. Note that I use an alpha value to make the plot more readable where there 
    # is a high density of points.
    plt.figure(figsize=(8, 6))
    plt.subplot(111)

    plt.scatter(Teff, Vabs, s=scaling*5, c=col, alpha=0.5)

    # Add an x and y label to the plot.
    plt.xlabel(r'$T_{\mathrm{eff}}\, [\mathrm{K}]$')
    plt.ylabel(r'$M_{\mathrm{V}}$')

    # Most stars have temperature <40000K and >1500 so
    # I put these as limits
    plt.xlim(1500, 40000)
    
    # And use a logarithmic x-axis.
    plt.xscale('log')
    
    
    # Invert the x and y-axes to follow astronomical convention
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    if includegrid:
        ax = plt.gca()
        yminorLocator = MultipleLocator(1.0)
        xmajorLocator = MultipleLocator(5000)
        ax.yaxis.set_minor_locator( yminorLocator )
        ax.xaxis.set_major_locator( xmajorLocator )
        plt.grid(which='major', color='#999999')

    # Finally, show the plot or save it as a PDF
    if pdf is not None:
        plt.savefig(pdf, format='pdf')
    else:
        plt.show()

    


def solar_system():
    """
    A simple function to return a list of Solar system planets/
    """

    # Here are the masses and radii in units of Earth masses and Earth radii. 
    # I multiply this by the mass/radius of the Earth in Jupiter masses
    m_earth_Jup = 5.972e24/1.89813e27
    r_earth_Jup = 6371./69911.
    m_planets = np.array([0.0553, 0.815, 1.0, 0.107, 317.83,
                          95.159, 14.536, 17.147])*m_earth_Jup
    r_planets = np.array([0.3829, 0.9499, 1.0, 0.532,
                          10.97, 9.14, 3.981, 3.865])*r_earth_Jup

    # Semi-major axis in AU
    a_planets = np.array([0.3870993, 0.723336, 1.000003,
                          1.52371, 5.2029, 9.537, 19.189, 30.0699])
    e_planets = np.array([0.20564, 0.00678, 0.01671, 0.09339,
                          0.0484, 0.0539, 0.04726, 0.00859])
    i_planets = np.array([7.01, 3.39, 0, 1.85, 1.31, 2.49, 0.77, 1.77])
    names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 
             'Uranus', 'Neptune']

    return {'mass': m_planets, 'radius': r_planets, 'a': a_planets, 
            'e': e_planets, 'i': i_planets, 'name': names}
    


def find_multiple_planetary_systems(t):
    """
    This routine finds all multiple planetary systems in a table. It does
    this by first finding all unique host star names and then going through each
    of these to find planets belonging to each system. 
    """

    n_all = len(t['name'])
    flag_multiple = np.zeros(n_all)
    
    uq = np.unique(t['star_name'])
    n_unique = len(uq)

    # I now do the next using a loop - it would be possible to avoid
    # this by using the index returned from unique but I thought this
    # was a bit more readable.
    systems = [] 
    max_extent = np.zeros(n_unique)

    for i in range(n_unique):
        # Find the stars with this name and insert the 
        # number of planets with this host star into flag_multiple. 
        planets_this, = np.where(t['star_name'] == uq[i])
        flag_multiple[i] = len(planets_this)

        # I get the maximum extent 
        max_extent[i] = np.max((t['semi_major_axis'][planets_this]))

        # Next, we will check that we have all information we 
        # want for this system. I write the logic in this order,
        # even though I then have an empty 'true' statement,
        # because I thought it was more readable.
        
        if (np.all(np.isfinite(t['mass'][planets_this]))
            and np.all(np.isfinite(t['radius'][planets_this]))
            and  np.all(np.isfinite(t['semi_major_axis'][planets_this]))
            and np.all(np.isfinite(t['star_radius'][planets_this]))):
            # Then we are ok.
            pass
        else:
            # Signal a problem by taking the negative.
            flag_multiple[i] = -flag_multiple[i]

        # Finally, add this extra-solar system to the systems list.
        # In here I also store the indices of the planets in the table
        # to easier work with the data afterwards.
        tmp = {'indices': planets_this, 'star_name': uq[i], 
               'extent': max_extent[i],
               'star_radius': t['star_radius'][planets_this[0]]}
        systems.append(tmp)
        
        
    # We want to return both the systems list, and the max_extent 
    # array (so the user can easily sort the system list)

    return (systems, flag_multiple, max_extent)
    

def compare_systems_to_solar_system(t, ps_all=None, flag_all=None,
                                    max_extent_all=None,
                                    n_min=2, scale_radius=10, xmax=None,
                                    pdf=None):
    """
    This function solves task 2d). The idea here is to compare the
    various multi-planet systems to our solar system. This routine plots
    the planetary systems on a line with the solar system in the middle. 

    Only planetary systems with a star with known properties as well
    as planet characterstics available are shown. They are ordered so that
    the most compact is on top. Here compactness is simply defined as the 
    radius of the outermost plantary orbit.
    """

    # Get the planetary systems and their maximum extents.
    if (ps_all is None or flag_all is None or max_extent_all is None):
        print "Finding the multiple planetary systems"
        ps_all, flag_all, max_extent_all = find_multiple_planetary_systems(t)
    
    
    # Find the planet system with >= n_min planets.
    use, = np.where(flag_all >= n_min)
    n_use = len(use)
    print "I found a total of {0} planetary systems with >= {1} planets".format(n_use, n_min)

    # Extract these parts for ease later
    ps_tmp = [ps_all[i] for i in use]
    flag = flag_all[use]
    max_extent = max_extent_all[use]

    # Now sort these on max_extent
    si = np.argsort(max_extent)
    max_extent = max_extent[si]
    ps = [ps_tmp[i] for i in si]
    flag = flag[si]

        
    # Get the solar system
    sol = solar_system()

    
    print "The length of the array={0}".format(len(flag))

    # Now set up the plot range if needed
    if xmax is None:
        xmax = np.max(max_extent)

    fig, ax = plt.subplots(n_use+1, 1, sharex=True)

    # And loop over each system and show them.
    for i in range(len(flag)):

        inds = ps[i]['indices']
        radii = t['radius'][inds]
        scale = radii*scale_radius

        # Note that I place the star at x=0 and have the 
        # planets to the left - so at negative x values.
        x_pos = -t['semi_major_axis'][inds]
        ax[i].scatter(0, 0, s=ps[i]['star_radius']*scale_radius,
                      marker='o', color='orange')
        ax[i].scatter(x_pos, x_pos*0, s=scale, marker='o')
        ax[i].set_xlim(-xmax, 0.5)
        ax[i].set_ylim(-0.5, 0.5)

        # Turn off the frame & axes
        ax[i].set_frame_on(False)
        ax[i].axes.get_yaxis().set_visible(False)
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].text(-xmax*0.95, 0.0, t['star_name'][inds[0]])
        
    # Finally, show the Solar system.
    i_sol = len(ax)-1
    scale = sol['radius']*scale_radius
    ax[i_sol].scatter(-sol['a'], sol['a']*0, s=scale, color='r',
                      marker='o')
    ax[i_sol].scatter(0, 0, s=scale_radius, color='yellow',
                      marker='o')
    ax[i_sol].set_frame_on(False)
    ax[i_sol].axes.get_yaxis().set_visible(False)
    ax[i_sol].spines['top'].set_visible(False)
    ax[i_sol].xaxis.set_ticks_position('bottom')
    ax[i_sol].plot([-xmax, 0], [-0.5, -0.5], color='black')
    ax[i_sol].set_xlabel('a [AU]')
    ax[i_sol].text(-xmax*0.95, 0.0, 'Solar system')


    if (pdf is not None):
        # In this case, save a PDF file
        fig.set_size_inches(7, 7*n_use/20.)
        fig.savefig(pdf, frameon=False)
    else:
        fig.show()
