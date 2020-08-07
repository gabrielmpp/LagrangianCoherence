"""
Module LCS
================

This module provides class (LCS) to compute the Finite-time Lyapunov Exponent from atmospheric wind fields (latitude, longitude
and time).
"""

from dask.diagnostics import ProgressBar
import xarray as xr
import numpy as np
from LagrangianCoherence.LCS.trajectory import parcel_propagation
from xr_tools.tools import latlonsel
from scipy.linalg import norm


class LCS:
    """
    API to compute the Finite-time Lyapunov exponent in 2D wind fields
    """
    earth_r = 6371000  # metres

    def __init__(self, timestep: float = 1, timedim='time',
                 SETTLS_order=0, subdomain=None, return_dpts=False, gauss_sigma=None):
        """

        :param timestep: float,
            Timestep length in seconds.
        :param dataarray_template:

        :param timedim: str,
            Name of the time dimension, default is 'time'.
        :param subtimes_len:
            Sub-intervals to divide the time integration, default is 1.

        ----------
        """
        self.timestep = timestep
        self.SETTLS_order = SETTLS_order
        self.timedim = timedim
        self.subdomain = subdomain
        self.gauss_sigma = gauss_sigma
        self.return_dpts = return_dpts
    def __call__(self, ds: xr.Dataset = None, u: xr.DataArray = None, v: xr.DataArray = None,
                 verbose=True, s=None, resample=None) -> xr.DataArray:

        """

        :param ds: xarray.Dataset, optional
            xarray dataset containing u and v as variables. Mutually exclusive with parameters u and v.
        :param u: xarray.Dataarray, optional
            xarray datarray containing u-wind component. Mutually exclusive with parameter ds.
        :param v: xarray.Dataarray, optional
            xarray datarray containing u-wind component. Mutually exclusive with parameter ds.
        :param verbose: bool, optional
            Whether to print intermediate values

        :return: xarray.Dataarray with the Finite-Time Lyapunov vector

        >>> subtimes_len = 1
        >>> timestep = -6*3600 # 6 hours in seconds
        >>> lcs = LCS(timestep=timestep, timedim='time', subtimes_len=subtimes_len)
        >>> ds = sampleData()
        >>> ftle = lcs(ds, verbose=False)
        """
        global verboseprint
        print('!' * 100)
        verboseprint = print if verbose else lambda *a, **k: None

        timestep = self.timestep
        timedim = self.timedim
        self.verbose = verbose

        if isinstance(ds, xr.Dataset):
            u = ds.u.copy()
            v = ds.v.copy()
        elif isinstance(ds, str):
            ds = xr.open_dataset(ds)
            u = ds.u.copy()
            v = ds.v.copy()
        if isinstance(resample, str):
            u = u.resample({self.timedim: resample}).interpolate('linear')
            v = v.resample({self.timedim: resample}).interpolate('linear')
            timestep = np.sign(timestep)*(u[self.timedim].values[1] - u[self.timedim].values[0]).astype('timedelta64[s]').astype('float')
        u_dims = u.dims
        v_dims = v.dims

        assert set(u_dims) == set(v_dims), "u and v dims are different"
        assert set(u_dims) == {'latitude', 'longitude', timedim}, 'array dims should be latitude and longitude only'

        verboseprint("*---- Parcel propagation ----*")
        x_departure, y_departure = parcel_propagation(u, v, timestep, propdim=self.timedim,
                                                      SETTLS_order=self.SETTLS_order,
                                                      verbose=verbose, s=s)

        verboseprint("*---- Computing deformation tensor ----*")

        def_tensor = compute_deftensor(x_departure, y_departure, sigma=self.gauss_sigma)
        if isinstance(self.subdomain, dict):
            def_tensor = latlonsel(def_tensor, **self.subdomain)
        def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        # eigenvalues = xr.apply_ufunc(lambda x: compute_eigenvalues(x), def_tensor.groupby('points'),
        #                             dask='parallelized',
        #                             output_dtypes=[float]
        #                             )
        verboseprint("*---- Computing eigenvalues ----*")
        vals = def_tensor.transpose(..., 'points').values
        vals = vals.reshape([2, 2, def_tensor.shape[-1]])
        def_tensor_norm = norm(vals, axis=(0, 1))
        def_tensor_norm = def_tensor.isel(derivatives=0).drop('derivatives').copy(data=def_tensor_norm)
        verboseprint("*---- Done eigenvalues ----*")
        def_tensor_norm = def_tensor_norm.unstack('points')
        timestamp = u[self.timedim].values[0] if np.sign(timestep) == 1 else u[self.timedim].values[-1]
        def_tensor_norm['time'] = timestamp
        eigenvalues = def_tensor_norm.expand_dims(self.timedim)
        if self.return_dpts:
            return eigenvalues, x_departure, y_departure
        else:
            return eigenvalues


def compute_deftensor(x_departure: xr.DataArray, y_departure: xr.DataArray, sigma=None) -> xr.DataArray:

    """
    Method to compute the deformation tensor

    Parameters
    ----------
    x_departure
    y_departure

    Returns
    -------
    """



    if isinstance(sigma, (float, int)):
        from scipy.ndimage import gaussian_filter
        x_departure = x_departure.copy(data=gaussian_filter(x_departure, sigma=sigma))
        y_departure = y_departure.copy(data=gaussian_filter(y_departure, sigma=sigma))
    # --- Conversion from Continuum Mechanics for Engineers: Theory and Problems, X Oliver, C Saracibar
    # Line element https://en.wikipedia.org/wiki/Spherical_coordinate_system
    earth_r = 6371000
    y =  y_departure.latitude.copy()
    y = y * np.pi / 180  # to rad
    x = x_departure.longitude.copy() * np.pi / 180
    X = x_departure.copy() * np.pi/180
    Y = y_departure.copy() * np.pi/180

    dX = earth_r * np.cos(Y) * X.diff('longitude')
    dx = earth_r * np.cos(y) * x.diff('longitude')

    dY = earth_r * Y.diff('latitude')
    dy = earth_r * y.diff('latitude')

    dXdx = dX/dx
    dXdy = dX/dy
    dYdy = dY/dy
    dYdx = dY/dx

    dXdx = dXdx.transpose('latitude', 'longitude')
    dXdy = dXdy.transpose('latitude', 'longitude')
    dYdy = dYdy.transpose('latitude', 'longitude')
    dYdx = dYdx.transpose('latitude', 'longitude')

    dXdx.name = 'dxdx'
    dXdy.name = 'dxdy'
    dYdy.name = 'dydy'
    dYdx.name = 'dydx'

    def_tensor = xr.merge([dXdx, dXdy, dYdx, dYdy])
    def_tensor = def_tensor.to_array()
    def_tensor = def_tensor.rename({'variable': 'derivatives'})
    def_tensor = def_tensor.transpose('derivatives', 'latitude', 'longitude')

    return def_tensor


def create_arrays_list(ds, groupdim='points'):
    ds_groups = list(ds.groupby(groupdim))
    input_arrays = []
    for label, group in ds_groups:
        input_arrays.append(group.values)
    return input_arrays


if __name__ == '__main__':
    import sys
    import subprocess
    # Args: timestep, timedim, SETTLS_order, subdomain, ds_path, outpath
    coords = str(sys.argv[4]).split('/')
    subdomain = {'longitude': slice(float(coords[0]), float(coords[1])),
                 'latitude': slice(float(coords[2]), float(coords[3]))}
    lcs = LCS(timestep=float(sys.argv[1]), timedim=str(sys.argv[2]), SETTLS_order=int(sys.argv[3]),
              subdomain=subdomain)
    input_path = str(sys.argv[5])
    out = lcs(ds=str(sys.argv[5]), s=1e5, resample='3H')
    out.to_netcdf(sys.argv[6])
    subprocess.call(['rm', input_path])
