from transforms3d.euler import euler2mat, mat2euler

def make_simple_data_4D():
    """
    This creates the simple data underlying the full set.
    
    Here I create a sinusoidal curve in the x-y plane, and add noise around it in two further dimensions. 
    I rotate the x, y, z dimensions and then output this as well as the original coordinates.
    
    """
    N = 500
    # Not centred on 0 and they have a different extent
    xrange = [3, 7]
    yrange = [2, 6]
    
    dz = 0.1
    dw = 0.1

    np.random.seed(100)
    x = np.random.uniform(xrange[0], xrange[1], N)
    y = 4*np.sin(x*np.pi*2) # np.random.uniform(yrange[0], yrange[1], N)
    w = np.random.normal(0, dw, size=N)
    z = np.random.normal(0, dz, size=N)
    
    # Create the rotation matrix using Euler angles.
    x_angle = -np.pi/3.
    y_angle = -np.pi / 5.2
    z_angle = np.pi/4.5
    R = euler2mat(x_angle, y_angle, z_angle, 'sxyz')
    
    X = np.vstack([x, y, z])
    Xnew = np.matmul(R, X)
    
    xr = Xnew[0, :].squeeze()
    yr = Xnew[1, :].squeeze()
    zr = Xnew[2, :].squeeze()
    Xr = np.vstack([xr, yr, zr, w])
    X = np.vstack([x, y, z, w])
    
    df = pd.DataFrame(Xr.T, columns=('X', 'Y', 'Z', 'W'))
    dfo = pd.DataFrame(X.T, columns=('X', 'Y', 'Z', 'W'))
    
    return df, dfo, Xr.T, X.T

