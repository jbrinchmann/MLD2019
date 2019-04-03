from __future__ import print_function
import utilities as DDM17
import pandas as pd
import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np

def read_visualisation_data():
    """
    Read in the visualisation data from the database using Pandas

    Returns
    -------
    table_names : list
       A list of table names (or data set names)
    data : dict
       A dictionary of tables. The top-level key is the name of the data
       set. (Set1, Set2 etc.). The value of this is a pandas table with
       keys x & y. So data['Set1'] is a pandas table for instance.xs


    """

    db = 'ThirteenDatasets.db'
    con = lite.connect(db)

    # Get the tables.
    rows = con.execute('SELECT name FROM sqlite_master WHERE type="table"')
    table_names = [row[0] for row in rows]

    # Now loop over these and create a dict with each set.
    data = dict()
    for tname in table_names:
        t = pd.read_sql_query("Select x, y From {0}".format(tname), con)
        data[tname] = t

    con.close()
        
    # I return both the list of table names and the
    # data dictionary mostly for convenience. 
    return table_names, data


def data_to_pandas_df(tnames, data):
    """Convert data dict to pandas data frame

    This routine takes the data dictionary from read_visualisation_data
    and creates a Pandas data frame with x & y as keys as well as a column
    with the dataset name

    Parameters
    ----------
    tnames : list
        A list of names of the tables to include. Normally taken to be 
        the output of read_visualisation_data
    data : dict
        A dict with the data, each dataset indexed by the table name.
        This is the format returned by read_visualisation_data
    """

    # Since the datasets here have the same length, I could
    # create the arrays first and then populate them - that
    # would be faster and I would avoid the asarray gymnastics
    # below to keep it as numpy arrays.
    #
    # But numpy append is fine for this.

    dataset = []
    first = True
    for i, tn in enumerate(tnames):
        xcoord = np.asarray(data[tn]['x'].data)
        ycoord = np.asarray(data[tn]['y'].data)
        if (first):
            x = xcoord.copy()
            y = ycoord.copy()
            first = False
        else:
            x = np.append(x, xcoord)
            y = np.append(y, ycoord)

        # Append the table name for each x value
        label = [tn for i in range(len(xcoord))]
        dataset = dataset + label

    df = pd.DataFrame({'x': x, 'y': y, 'set': dataset})

    return df
        
def show_visualisations_multipanel(tnames, data):
    """
    Show 2D visualisations for the data

    Parameters
    ----------
    tnames : list
        A list of names of the tables to plot. Normally taken to be 
        the output of read_visualisation_data
    data : dict
        A dict with the data, each dataset indexed by the table name.
        This is the format returned by read_visualisation_data
    
    """

    nx = 5
    ny = 3
    fig, axes = plt.subplots(ncols=nx, nrows=ny, figsize=(12, 8),
                                 sharex=True, sharey=True)
    plt.tick_params(axis='both', which='major', labelsize=8)
    dims = axes.shape
    print("Axes shape=", dims)
    for i, tn in enumerate(tnames):
        x, y = data[tn]['x'], data[tn]['y']

        i_x, i_y = np.unravel_index(i, dims)
        axes[i_x, i_y].scatter(x, y, 10)
        axes[i_x, i_y].text(0.95, 0.93, tn, 
            transform=axes[i_x, i_y].transAxes, ha='right', va='top')
        axes[i_x, i_y].set_xlim(0, 120)
        axes[i_x, i_y].set_ylim(0, 120)
        
    axes[2, 3].set_axis_off()
    axes[2, 4].set_axis_off()

    # I also want to remove white-space between the panels
    fig.subplots_adjust(hspace=0)
    plt.show()


def get_statistics(tnames, data):
    """Calculate basic statistics for the data

    The data is assumed to be a dictionary returned from read_visualisation_data() 
    """

    # I know the number of data sets so I can make a set
    # of result arrays. There are more elegant ways to do this
    # for instance by using the .description function of Pandas
    # DataFrames, but I thought this was clearer.
    #
    # In general this is a less good solution though - because it
    # hardcodes information in several places
    n_datasets = len(tnames)
    stats = {'x': {'mean': np.zeros(n_datasets),
                       'median': np.zeros(n_datasets),
                       'std': np.zeros(n_datasets),
                       '25%': np.zeros(n_datasets),
                       '75%': np.zeros(n_datasets),
                       'max': np.zeros(n_datasets),
                       'min': np.zeros(n_datasets)},
            'y': {'mean': np.zeros(n_datasets),
                       'median': np.zeros(n_datasets),
                       'std': np.zeros(n_datasets),
                       '25%': np.zeros(n_datasets),
                       '75%': np.zeros(n_datasets),
                       'max': np.zeros(n_datasets),
                       'min': np.zeros(n_datasets)}}

    for i, tn in enumerate(tnames):
        for key in ('x', 'y'):
            var = data[tn][key]
            stats[key]['mean'][i] = np.mean(var)
            stats[key]['median'][i] = np.median(var)
            stats[key]['std'][i] = np.std(var)
            stats[key]['25%'][i] = np.percentile(var, 25.)
            stats[key]['75%'][i] = np.percentile(var, 75.)
            stats[key]['min'][i] = np.min(var)
            stats[key]['max'][i] = np.max(var)
            

    return stats

def get_statistics_compact(tnames, data):
    """
    Equivalent to the above - shorter and a bit more flexible but
    also probably a bit more less clear?
    """

    stats = {'x': dict(), 'y': dict()}

    first = True
    for tn in tnames:
        summary = data[tn].describe()

        # If we are doing the first round through we need to create
        # the lists
        todo = summary['x'].keys()
        for key in ('x', 'y'):
            for x_todo in todo:
                if first:
                    stats[key][x_todo] = []
                stats[key][x_todo].append(summary[key][x_todo])
                
        first = False

    return stats


def get_statistics_extendable(tnames, data, functions=None):
    """
    Equivalent to the above two, but this one is more extendible

    The functions argument should be an array of functions to
    apply to the data. If this is set to None the a default set 
    of functions are applied, namely:

        functions = {'mean': np.mean, 'median': np.median,
                         '25%': lambda x: np.percentile(x, 25.0),
                         '75%': lambda x: np.percentile(x, 75.0)}
    """

    stats = {'x': dict(), 'y': dict()}
    if functions is None:
        functions = {'mean': np.mean, 'median': np.median,
                         '25%': lambda x: np.percentile(x, 25.0),
                         '75%': lambda x: np.percentile(x, 75.0)}
    n_datasets = len(tnames)
    first = True
    for i, tn in enumerate(tnames):
        for key in ('x', 'y'):
            var = data[tn][key]
            
            for todo, func in functions.iteritems():
                if first:
                    stats[key][todo] = np.zeros(n_datasets)
                stats[key][todo][i] = func(var)
                
        first = False

    return stats


if __name__ == "__main__":
    # execute only if run as a script
    tnames, data = read_visualisation_data()

    show_visualisations_multipanel(tnames, data)
