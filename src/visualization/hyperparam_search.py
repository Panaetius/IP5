import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf

def main():
    # Generate Data...
    numdata = 100
    x = np.array([0.0023, 0.00011, 0.000094, 0.0061, 0.000036, 0.00016, 0.0078, 0.0013, 0.000043, 0.00001, 0.00001, 0.01, 0.000001, 0.01, 0.000037, 0.000017, 0.0000039, 0.0009, 0.0014, 0.000056, 0.000001, 0.00031, 0.0057])
    y = np.array([0.00062, 0.0019, 0.00065, 0.0002, 0.0073, 0.000016, 0.00012, 0.00017, 0.000031, 0.00001, 0.01, 0.00001, 0.01, 0.01, 0.005, 0.00023, 0.00014, 0.0012, 0.0047, 0.00057, 0.00001, 0.00011, 0.00097])
    z = np.array([3.5, 3.2, 3.1, 3.46, 3.279, 3.18, 3.44, 3.26, 3.32, 3.478, 3.496, 3.306, 4.06, 3.748, 3.252, 3.435, 3.815, 3.439, 3.411, 3.122, 4.023, 3.259, 3.45])
    z2 = 1 - np.array([0.067, 0.126, 0.13, 0.062, 0.13, 0.11, 0.04, 0.1, 0.12, 0.13, 0.093, 0.0879, 0.0309, 0.038, 0.131, 0.0775, 0.05348, 0.07273, 0.07754, 0.1238, 0.0303, 0.09697, 0.06417])
    

    # Fit a 3rd order, 2d polynomial
    #m = polyfit2d(x,y,z, 5)

    # Evaluate it on a grid...
    nx, ny = 100, 100
    xx, yy = np.meshgrid(np.logspace(-6, -2, nx), 
                         np.logspace(-5, -2, ny))

    points = np.column_stack((x,y))
    zz = griddata(points, z, (xx, yy), method='linear', fill_value=np.max(z), rescale=True)
    zz[(zz < np.min(z))] = np.min(z)
    zz[(zz > np.max(z))] = np.max(z)
    zz2 = griddata(points, z2, (xx, yy), method='linear', fill_value=np.max(z2))
    zz2[(zz2 < np.min(z2))] = np.min(z2)
    zz2[(zz2 > np.max(z2))] = np.max(z2)
    
    #rbf = Rbf(x,y,z)
    #zz =rbf(xx, yy)
    
    #zz = polyval2d(xx, yy, m)

    # Plot
    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    plt.title('Cross Entropy')

    axes = plt.gca()
    axes.set_xlim([0.000001, 0.01])
    axes.set_ylim([0.00001, 0.01])
    #plt.imshow(zz, extent=(0.000001, 0.01, 0.000001, 0.01))
    mm = plt.pcolor(xx, yy, zz)
    plt.contour(xx, yy, zz, 10, linewidths=1, colors='k')
    cbar = plt.colorbar(mm, ticks=[np.min(zz), np.max(zz)])
    cbar.ax.set_yticklabels([str(np.min(zz)), str(np.max(zz))])
    #plt.imshow(grid_z2.T, extent=(0.000001, 0.01, 0.000001, 0.01), origin='lower')
    plt.scatter(x, y, c=z)
    plt.yscale('log')
    plt.xscale('log')

    plt.subplot(212)
    plt.title('Inverse Accuracy')
    axes = plt.gca()
    axes.set_xlim([0.000001, 0.01])
    axes.set_ylim([0.00001, 0.01])
    #plt.imshow(zz, extent=(0.000001, 0.01, 0.000001, 0.01))
    mm = plt.pcolor(xx, yy, zz2)
    plt.contour(xx, yy, zz2, 10, linewidths=1, colors='k')
    cbar = plt.colorbar(mm, ticks=[np.min(zz2), np.max(zz2)])
    cbar.ax.set_yticklabels([str(np.min(zz2)), str(np.max(zz2))])
    #plt.imshow(grid_z2.T, extent=(0.000001, 0.01, 0.000001, 0.01), origin='lower')
    plt.scatter(x, y, c=z2)
    plt.yscale('log')
    plt.xscale('log')

    plt.show()

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()

