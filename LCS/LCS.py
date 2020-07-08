"""
Module LCS
================

This module provides class (LCS) to compute the Finite-time Lyapunov Exponent from atmospheric wind fields (latitude, longitude
and time).
"""
from scipy.linalg import eigvals

import xarray as xr
import numpy as np
from numba import jit
import numba
import pandas as pd
from typing import List
from LagrangianCoherence.LCS.trajectory import parcel_propagation
from xr_tools.tools import latlonsel
# Types of Lagrangian coherence:
LCS_TYPES: List[str]
LCS_TYPES = ['attracting', 'repelling']


class LCS:
    """
    API to compute the Finite-time Lyapunov exponent in 2D wind fields
    """
    earth_r = 6371000

    def __init__(self, lcs_type: str, timestep: float = 1, timedim='time',
                 shearless=False, SETTLS_order=0, subdomain=None, cg_lambda=np.max):
        """

        :param lcs_type: str,
            Type of coherent structure: 'attracting' or 'repelling'.
        :param timestep: float,
            Timestep length in seconds.
        :param dataarray_template:

        :param timedim: str,
            Name of the time dimension, default is 'time'.
        :param shearless: bool,
            Whether to ignore the shear deformation, default is False.
        :param subtimes_len:
            Sub-intervals to divide the time integration, default is 1.
        """
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.SETTLS_order = SETTLS_order
        self.timedim = timedim
        self.shearless = shearless
        self.subdomain = subdomain
        self.cg_lambda = cg_lambda

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
        >>> lcs = LCS(lcs_type='repelling', timestep=timestep, timedim='time', subtimes_len=subtimes_len)
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

        u_dims = u.dims
        v_dims = v.dims

        assert set(u_dims) == set(v_dims), "u and v dims are different"
        assert set(u_dims) == {'latitude', 'longitude', timedim}, 'array dims should be latitude and longitude only'

        # if not (hasattr(u, "x") and hasattr(u, "y")):
        #     verboseprint("Ascribing x and y coords do u")
        #     u = to_cartesian(u)
        # if not (hasattr(v, "x") and hasattr(v, "y")):
        #     verboseprint("Ascribing x and y coords do v")
        #     v = to_cartesian(v)

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

        eigenvalues = xr.apply_ufunc(self.compute_eigenvalues, def_tensor.groupby('points'),
                              input_core_dims=[['derivatives']])
        verboseprint("*---- Done eigenvalues ----*")
        eigenvalues = eigenvalues.unstack('points')
        eigenvalues = eigenvalues.expand_dims({self.timedim: [u[self.timedim].values[0]]})

        return eigenvalues

    def compute_eigenvalues(self, def_tensor):
        """

        :rtype: np.array
        """
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)
        eigenvalues = self.cg_lambda(np.real(eigvals(cauchy_green.reshape([2, 2]))))

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


