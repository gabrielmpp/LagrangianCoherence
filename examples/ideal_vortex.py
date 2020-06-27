import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def ideal_vortex(nx, ny):
    """
    Method to initialize an ideal vortex
    Parameters
    ----------
    nx grid width
    ny grid height

    Returns: vorticity numpy array
    -------
    """
    center_y = int(ny / 2)
    center_x = int(nx / 2)
    intensity = max(nx, ny)

    streamfunction = np.zeros([nx, ny])

    for x in range(nx):
        for y in range(ny):
            streamfunction[x, y] = intensity + (
                    (x - center_x)**2 + np.abs(y - center_y)**2
            ) ** 0.5
    streamgradient = np.gradient(streamfunction)
    u = streamgradient[1]
    v = -streamgradient[0]

    vorticity = np.gradient(streamgradient[0], axis=0) + np.gradient(streamgradient[0], axis=1) + \
                np.gradient(streamgradient[0], axis=0) + np.gradient(streamgradient[0], axis=1)

    return vorticity, streamfunction, u, v


nx, ny = 100, 100
vorticity, streamfunction, u, v = ideal_vortex(nx, ny)
vorticity = xr.DataArray(vorticity, dims=['x', 'y'], coords={'x': np.arange(nx),
                                                             'y': np.arange(ny)})
streamfunction = xr.DataArray(streamfunction, dims=['x', 'y'], coords={'x': np.arange(nx),
                                                             'y': np.arange(ny)})
streamfunction.plot()
plt.streamplot(vorticity.x.values, vorticity.y.values, u, v)
plt.show()