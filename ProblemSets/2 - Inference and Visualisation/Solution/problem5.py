#
# Run k-nearest neighbour.
#

import numpy as np
# import scipy.spatial.distance as dist
import utilities as u

def find_k_closest(X_i, X, k=3):
    """
    Find the k-closest sources
    """

    # Get the pairwise distances - I use squared distances to speed
    # things marginally up
    dd = (X_i-X)**2

    # Sort the distances.
    i_sort = np.argsort(dd.squeeze())

    # Find the k closest
    k_closest = i_sort[0:k]

#    print "Distances = ", dd
#    print "The closest ", k_closest

    return k_closest


def knn_regress(x, y, xout=None, k=3):
    """
    Calculate k-nearest neighbour regression for x vs y
    """

    if xout is None:
        xout = x

    # Our estimates
    yhat = np.zeros(len(xout))
    X = np.vstack([[x], [y]])
    for i, xo_i in enumerate(xout):
        i_close = find_k_closest(xo_i, x, k=k)

        yhat[i] = np.mean(y[i_close])

    return yhat


        
if __name__ == "__main__":
    # Get pickled data (you might need to change the path!)
    import matplotlib.pyplot as plt
    
    d = u.pickle_from_file('../points_example1.pkl')

    x, y_true, y, sigma = d['retval']

    xout = np.linspace(-3, 3, 100)

    yest1 = knn_regress(x,y, xout=xout, k=1)
    yest5 = knn_regress(x,y, xout=xout, k=5)
    yest9 = knn_regress(x,y, xout=xout, k=9)


    plt.plot(xout, yest1, color='red', label='k=1')
    plt.plot(xout, yest5, color='blue', label='k=5')
    plt.plot(xout, yest9, color='green', label='k=9')

    plt.scatter(x, y, color='gray', label='data')

    plt.legend()
    plt.show()
