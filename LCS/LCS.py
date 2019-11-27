"""
Module LCS
================

This module provides class (LCS) to compute the Finite-time Lyapunov Exponent from atmospheric wind fields (latitude, longitude
and time).K
"""

import xarray as xr
import numpy as np
from numba import jit
import numba
from typing import List

# Types of Lagrangian coherence:
LCS_TYPES: List[str]
LCS_TYPES = ['attracting', 'repelling']


class LCS:
    """
    API to compute the Finite-time Lyapunov exponent in 2D wind fields
    """
    earth_r = 6371000

    def __init__(self, lcs_type: str, timestep: float = 1, timedim='time',
                 shearless=False, subtimes_len=1):
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
        self.subtimes_len = subtimes_len
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.timedim = timedim
        self.shearless = shearless

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

        verboseprint("*---- Computing deformation tensor ----*")
        def_tensor = compute_deformation_tensor(u, v, timestep, verbose=verbose)
        def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        # eigenvalues = xr.apply_ufunc(lambda x: compute_eigenvalues(x), def_tensor.groupby('points'),
        #                             dask='parallelized',
        #                             output_dtypes=[float]
        #                             )
        input_arrays = create_arrays_list(def_tensor)
        verboseprint("*---- Computing eigenvalues ----*")
        data = compute_eigenvalues(input_arrays)
        verboseprint("*---- Done eigenvalues ----*")
        eigenvalues = def_tensor.copy(data=data)
        eigenvalues = eigenvalues.unstack('points')
        eigenvalues = eigenvalues.isel(derivatives=0).drop('derivatives')
        eigenvalues = eigenvalues.expand_dims({self.timedim: [u[self.timedim].values[0]]})

        return eigenvalues


def compute_deformation_tensor(u: xr.DataArray, v: xr.DataArray, timestep: float,
                               verbose=False) -> xr.DataArray:
    """
    :param u: xr.DataArray, array corresponding to the zonal wind field
    :param v: xr.DataArray, array corresponding to the meridional wind field
    :param timestep: float
    :return: the deformation tensor
    :rtype: xarray.Dataarray
    """

    x_departure, y_departure = parcel_propagation(u, v, timestep, propdim=self.timedim,
                                                  subtimes_len=self.subtimes_len, verbose=verbose)
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
    # from xrviz.dashboard import Dashboard
    # dashboard = Dashboard(def_tensor)
    # dashboard.show()
    return def_tensor


def parcel_propagation(U: xr.DataArray, V: xr.DataArray, timestep: int, propdim: str = "time",
                       verbose: bool = True, subtimes_len: int = 10, s: float = 1e5, return_traj: bool = False):
    """
    Method to propagate the parcel given u and v

    :param return_traj: boolean, default is False,
        True returns the parcel positions for all timesteps in a time dimension
    :param propdim: str,
        dimension name for time, default is 'time'
    :param timestep: float,
        size of time interval in seconds
    :param U: xarray.DataArray,
        Array corresponding the zonal wind
    :param V: xarray.Dataarray,
        Array correponding to the meridional wind
    :param s: float,
        smoothing factor for the spline spherical interpolation, default is 1e5
    :return: tuple,
        zonal and meridional arrays corresponding to the final positions of the trajectories
    """
    verboseprint = print if verbose else lambda *a, **k: None
    U = U.sortby('longitude')
    V = V.sortby('longitude')
    U = U.sortby('latitude')
    V = V.sortby('latitude')
    u_lat_values = (90 + U.latitude.values.copy()) * np.pi / 180
    u_lon_values = (180 + U.longitude.values.copy()) * np.pi / 180
    v_lat_values = (90 + V.latitude.values.copy()) * np.pi / 180
    v_lon_values = (180 + V.longitude.values.copy()) * np.pi / 180


    from scipy.interpolate import RectSphereBivariateSpline
    earth_r = 6371000
    conversion_y = 1 / earth_r  # converting m/s to rad/s
    conversion_x = 1 / (earth_r * xr.apply_ufunc(lambda x: np.abs(np.cos(x - 0.5 * np.pi)), U.latitude))
    conversion_x, _ = xr.broadcast(conversion_x, U.isel({propdim: 0}))
    times = U[propdim].values.tolist()
    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating

    positions_y, positions_x = np.meshgrid(u_lat_values.copy(), u_lon_values.copy())
    # positions_y

    initial_pos = xr.DataArray()
    pos_list_x = []
    pos_list_y = []
    for time in times:
        verboseprint(f'Propagating time {time}')
        subtimes = np.arange(0, subtimes_len, 1).tolist()

        for subtime in subtimes:
            verboseprint(f'Propagating subtime {subtime}')

            subtimestep = timestep / subtimes_len

            lat = positions_y[0, :].copy()  # lat is constant along cols
            lon = positions_x[:, 0].copy()  # lon is constant along rows
            # ---- propagating positions ---- #
            v_data = V.sel({propdim: time}).values
            interpolator_y = RectSphereBivariateSpline(v_lat_values, v_lon_values, v_data, s=s)

            positions_y = positions_y + \
                          subtimestep * conversion_y * \
                          interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            # Hard boundary
            positions_y[np.where(positions_y < 0)] = 0
            positions_y[np.where(positions_y > np.pi)] = np.pi

            u_data = U.sel({propdim: time}).values
            interpolator_x = RectSphereBivariateSpline(u_lat_values, u_lon_values, u_data)
            positions_x = positions_x + \
                          subtimestep * conversion_x.values.T * \
                          interpolator_x.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

            # Hard boundary
            positions_x[np.where(positions_x < 0)] = 0
            positions_x[np.where(positions_x > 2 * np.pi)] = 2 * np.pi
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)

    for i in range(len(pos_list_x)):
        pos_list_x[i] = pos_list_x[i] * 180 / np.pi - 180
        pos_list_y[i] = pos_list_y[i] * 80 / np.pi - 90
        pos_list_x[i] = xr.DataArray(pos_list_x[i].T, dims=['latitude', 'longitude'],
                               coords=[U.latitude.values.copy(), U.longitude.values.copy()])
        pos_list_y[i] = xr.DataArray(pos_list_y[i].T, dims=['latitude', 'longitude'],
                               coords=[U.latitude.values.copy(), U.longitude.values.copy()])
    if return_traj:
        positions_x = xr.concat(pos_list_x, dim=U[propdim])
        positions_y = xr.concat(pos_list_y, dim=U[propdim])
    else:
        positions_x = pos_list_x[-1]
        positions_y = positions_y[-1]

    return positions_x, positions_y


@jit(parallel=True)
def compute_eigenvalues(arrays_list):
    """

    :rtype: np.array
    """
    out_list = []
    for i in numba.prange(len(arrays_list)):
        def_tensor = arrays_list[i]
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)
        eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        eigenvalues = np.repeat(eigenvalues, 4).reshape(
            [4])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        out_list.append(eigenvalues)
    out = np.stack(out_list, axis=1)
    return out


@jit(parallel=True)
def create_arrays_list(ds, groupdim='points'):
    ds_groups = list(ds.groupby(groupdim))
    input_arrays = []
    for label, group in ds_groups:  # have to do that because bloody groupby returns the labels
        input_arrays.append(group.values)
    return input_arrays


def sampleData() -> xr.Dataset:
    """
    Function to create sample xr.Dataarray
    :return: xr.Dataarray

    >>> sampleData()
    <xarray.Dataset>
    Dimensions:    (latitude: 200, longitude: 100, time: 4)
    Coordinates:
      * latitude   (latitude) float64 -80.0 -79.5 -78.99 -78.49 ... 18.99 19.5 20.0
      * longitude  (longitude) float64 -80.0 -79.49 -78.99 ... -31.01 -30.51 -30.0
      * time       (time) datetime64[ns] 2000-01-01 ... 2000-01-01T18:00:00
    Data variables:
        u          (latitude, longitude, time) float64 19.4 18.79 ... 2.053 1.712
        v          (latitude, longitude, time) float64 4.679 4.534 ... 26.33 21.95

    """
    import pandas as pd
    ntime = 4
    ky = 10
    kx = 40
    lat1 = -80
    lat2 = 20
    lon1 = -80
    lon2 = -30
    dx = 0.5

    nlat = int((lat2 - lat1) / dx)
    nlon = int((lon2 - lon1) / dx)
    latitude = np.linspace(lat1, lat2, nlat)
    longitude = np.linspace(lon1, lon2, nlon)
    time = pd.date_range("2000-01-01T00:00:00", periods=ntime, freq="6H")
    time_idx = np.array([x for x in range(len(time))])

    frq = 0.25
    u_data = 20 * np.ones([nlat, nlon, len(time)]) * (np.sin(ky * np.pi * latitude / 180) ** 2).reshape(
        [nlat, 1, 1]) * np.cos(time_idx * frq).reshape([1, 1, len(time)])
    v_data = 40 * np.ones([nlat, nlon, len(time)]) * (np.sin(kx * np.pi * longitude / 360) ** 2).reshape(
        [1, nlon, 1]) * np.cos(time_idx * frq).reshape([1, 1, len(time)])

    u = xr.DataArray(u_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude.copy(), 'longitude': longitude.copy(), 'time': time.copy()})
    v = xr.DataArray(v_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude.copy(), 'longitude': longitude.copy(), 'time': time.copy()})
    u.name = 'u'
    v.name = 'v'
    ds = xr.merge([u, v])
    return ds


def run_example():
    import matplotlib.pyplot as plt
    import pandas as pd

    ftle = lcs(ds)
    dep_x, dep_y = parcel_propagation(u.copy(), v.copy(), timestep=timestep, subtimes_len=subtimes_len)
    origin = np.meshgrid(longitude, latitude)[1]
    displacement = dep_x.copy(data=dep_y - origin)
    mag = (u.isel(time=0).values ** 2 + v.isel(time=0).values ** 2) ** 0.5
    plt.streamplot(longitude, latitude, u.isel(time=0).values, v.isel(time=0).values, color=mag)
    (displacement / len(time)).plot(vmax=10, vmin=-10, cmap="RdBu")
    plt.show()
    dep_x.plot(cmap='rainbow', vmin=-80, vmax=-30)
    plt.show()
    dep_y.plot(cmap='rainbow', vmax=20, vmin=-80)
    plt.streamplot(longitude, latitude, u.isel(time=0).values, v.isel(time=0).values)
    plt.show()

    #
    u.isel(time=0).plot()
    plt.show()
    plt.streamplot(longitude, latitude, u.isel(time=2).values, v.isel(time=2).values)
    ftle = ftle.where(dep_x <= dep_x.longitude.max())
    ftle = ftle.where(dep_x >= dep_x.longitude.min())
    ftle = ftle.where(dep_y <= dep_x.latitude.max())
    ftle = ftle.where(dep_y >= dep_x.latitude.min())

    ftle.isel(time=0).plot(vmax=50)
    plt.streamplot(longitude, latitude, u.isel(time=0).values, v.isel(time=0).values)

    plt.show()
    print("s")


if __name__ == '__main__':
    run_example()
