"""
Module LCS
================

This module provides class (LCS) to compute the Finite-time Lyapunov Exponent from atmospheric wind fields (latitude, longitude
and time).
"""

from scipy.linalg import eigvals
from dask.diagnostics import ProgressBar
import xarray as xr
import numpy as np
from numba import jit
import numba
import pandas as pd
from typing import List
from LagrangianCoherence.LCS.trajectory import parcel_propagation
from xr_tools.tools import latlonsel


class LCS:
    """
    API to compute the Finite-time Lyapunov exponent in 2D wind fields
    """
    earth_r = 6371000  # metres

    def __init__(self, timestep: float = 1, timedim='time',
                 SETTLS_order=0, subdomain=None):
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

    def __call__(self, ds: xr.Dataset = None, u: xr.DataArray = None, v: xr.DataArray = None,
                 verbose=True) -> xr.DataArray:

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

        u_dims = u.dims
        v_dims = v.dims

        assert set(u_dims) == set(v_dims), "u and v dims are different"
        assert set(u_dims) == {'latitude', 'longitude', timedim}, 'array dims should be latitude and longitude only'

        verboseprint("*---- Parcel propagation ----*")
        x_departure, y_departure = parcel_propagation(u, v, timestep, propdim=self.timedim,
                                                      SETTLS_order=self.SETTLS_order,
                                                      verbose=verbose)

        verboseprint("*---- Computing deformation tensor ----*")

        def_tensor = compute_deformation_tensor(x_departure, y_departure)
        if isinstance(self.subdomain, dict):
            def_tensor = latlonsel(def_tensor, **self.subdomain)
        def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        # eigenvalues = xr.apply_ufunc(lambda x: compute_eigenvalues(x), def_tensor.groupby('points'),
        #                             dask='parallelized',
        #                             output_dtypes=[float]
        #                             )
        verboseprint("*---- Computing eigenvalues ----*")

        # def_tensor = def_tensor.chunk({'points': int(def_tensor.points.shape[0]/10)})
        def_tensor = def_tensor.chunk({'points': 1})

        # -- Observation: Numpy's norm is equivalent to the square-root of the Cauchy-Green strain tensor.

        eigenvalues = xr.apply_ufunc(lambda x: np.array(np.linalg.norm(x.reshape([2, 2]))).reshape([1]),
                                     def_tensor,
                                     input_core_dims=[['derivatives']],
                                     # exclude_dims=set(('derivatives')),
                                     # output_core_dims=[['derivatives']],
                                     dask='parallelized',
                                     output_dtypes=[float])

        with ProgressBar():
            eigenvalues = eigenvalues.load()

        verboseprint("*---- Done eigenvalues ----*")
        eigenvalues = eigenvalues.unstack('points')
        timestamp = u[self.timedim].values[0] if np.sign(timestep) == 1 else u[self.timedim].values[-1]
        eigenvalues['time'] = timestamp
        eigenvalues = eigenvalues.expand_dims(self.timedim)

        return eigenvalues


def compute_deformation_tensor(x_departure: xr.DataArray, y_departure: xr.DataArray) -> xr.DataArray:
    """
    :param u: xr.DataArray, array corresponding to the zonal wind field
    :param v: xr.DataArray, array corresponding to the meridional wind field
    :param timestep: float
    :return: the deformation tensor
    :rtype: xarray.Dataarray
    """

    # u, v, eigengrid = interpolate_c_stagger(u, v)
    conversion_dydx = xr.apply_ufunc(lambda x: np.cos(x * np.pi / 180), y_departure.latitude)
    conversion_dxdy = xr.apply_ufunc(lambda x: np.cos(x * np.pi / 180), x_departure.latitude)

    dxdx = x_departure.diff('longitude') / x_departure.longitude.diff('longitude')
    dxdy = conversion_dxdy * x_departure.diff('latitude') / x_departure.latitude.diff('latitude')
    dydy = y_departure.diff('latitude') / y_departure.latitude.diff('latitude')
    dydx = y_departure.diff('longitude') / (y_departure.longitude.diff('longitude') * conversion_dydx)

    dxdx = dxdx.transpose('latitude', 'longitude')
    dxdy = dxdy.transpose('latitude', 'longitude')
    dydy = dydy.transpose('latitude', 'longitude')
    dydx = dydx.transpose('latitude', 'longitude')

    dxdx.name = 'dxdx'
    dxdy.name = 'dxdy'
    dydy.name = 'dydy'
    dydx.name = 'dydx'

    def_tensor = xr.merge([dxdx, dxdy, dydx, dydy])
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
    out = lcs(ds = str(sys.argv[5]))
    out.to_netcdf(sys.argv[6])
    subprocess.call(['rm', input_path])
