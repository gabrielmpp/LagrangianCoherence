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
from LagrangianCoherence.LCS.tools import derivative_spherical_coords, fourth_order_derivative
from IPython.core.debugger import set_trace


class LCS:
    """
    API to compute the Finite-time Lyapunov exponent in 2D wind fields
    """
    earth_r = 6371000  # metres

    def __init__(self, timestep: float = 1, timedim='time',
                 SETTLS_order=0, subdomain=None, return_dpts=False,
                 gauss_sigma=None):
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
                 verbose=True, s=None, resample=None, s_is_error=False, isglobal=False,
                 return_traj=False, interp_to_common_grid=True,
                 traj_interp_order=3) -> xr.DataArray:

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

        #  Ascending order is required
        #  TODO: Add checks for continuity

        u = u.sortby('latitude')
        u = u.sortby('longitude')
        v = v.sortby('latitude')
        v = v.sortby('longitude')
        if isglobal:
            if interp_to_common_grid:
                lats = np.linspace(-89.75, 89.75, 180 * 2)
                lons = np.linspace(-180, 179.5, 360 * 2 + 1)
                u_reindex = u.reindex(latitude=lats, longitude=lons, method='nearest')
                v_reindex = v.reindex(latitude=lats, longitude=lons, method='nearest')
                u_interp = u.interp(latitude=lats, longitude=lons, method='linear')
                v_interp = v.interp(latitude=lats, longitude=lons, method='linear')
                u = u_interp.where(~xr.ufuncs.isnan(u_interp), u_reindex)
                v = v_interp.where(~xr.ufuncs.isnan(v_interp), v_reindex)
            cyclic_xboundary = True
            self.subdomain = None
        else:
            cyclic_xboundary = False
        if s is None:
            s = int(10*u.isel({timedim: 0}).size * u.isel({timedim: 0}).std())
            print(f'using s = ' + str(s/1e6) + '1e6')
        verboseprint("*---- Parcel propagation ----*")
        x_departure, y_departure = parcel_propagation(u, v, timestep, propdim=self.timedim,
                                                      SETTLS_order=self.SETTLS_order,
                                                      verbose=verbose,
                                                      cyclic_xboundary=cyclic_xboundary, return_traj=return_traj,
                                                      interp_order=traj_interp_order)
        if return_traj:
            x_trajs = x_departure.copy()
            y_trajs = y_departure.copy()
            x_departure = x_departure.isel({timedim: -1})
            y_departure = y_departure.isel({timedim: -1})
        verboseprint("*---- Computing deformation tensor ----*")

        def_tensor = flowmap_gradient(x_departure, y_departure, sigma=self.gauss_sigma)
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
        vals = vals.reshape([3, 3, def_tensor.shape[-1]])
        def_tensor_norm = norm(vals, axis=(0, 1), ord=2)
        def_tensor_norm = def_tensor.isel(derivatives=0).drop('derivatives').copy(data=def_tensor_norm)
        verboseprint("*---- Done eigenvalues ----*")
        def_tensor_norm = def_tensor_norm.unstack('points')
        timestamp = u[self.timedim].values[0] if np.sign(timestep) == 1 else u[self.timedim].values[-1]
        def_tensor_norm['time'] = timestamp
        eigenvalues = def_tensor_norm.expand_dims(self.timedim)
        if self.return_dpts and return_traj:
            return eigenvalues, x_departure, y_departure, x_trajs, y_trajs
        elif self.return_dpts:
            return eigenvalues, x_departure, y_departure
        elif return_traj:
            return eigenvalues, x_trajs, y_trajs
        else:
            return eigenvalues


def flowmap_gradient(x_departure: xr.DataArray, y_departure: xr.DataArray, sigma=None) -> xr.DataArray:

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
    model_res = .25 * np.pi/180
    LON = x_departure.copy() * np.pi/180
    LAT = (y_departure.copy()-90) * np.pi/180  # colatitude in radians
    X = earth_r * np.sin(LAT) * np.cos(LON)
    Y = earth_r * np.sin(LAT) * np.sin(LON)
    Z = earth_r * np.cos(LAT)
    dXdx = derivative_spherical_coords(X, dim=1)
    dXdy = derivative_spherical_coords(X, dim=0)
    dYdx = derivative_spherical_coords(Y, dim=1)
    dYdy = derivative_spherical_coords(Y, dim=0)
    dZdx = derivative_spherical_coords(Z, dim=1)
    dZdy = derivative_spherical_coords(Z, dim=0)
    dXdr = xr.zeros_like(dXdx)
    dYdr = xr.zeros_like(dYdx)
    dZdr = xr.zeros_like(dZdx)

    dXdx.name = 'dxdx'
    dXdy.name = 'dxdy'
    dYdy.name = 'dydy'
    dYdx.name = 'dydx'
    dZdx.name = 'dzdx'
    dZdy.name = 'dzdy'
    dXdr.name = 'dxdr'
    dYdr.name = 'dydr'
    dZdr.name = 'dzdr'

    def_tensor = xr.merge([dXdx, dXdy, dYdx, dYdy, dZdx, dZdy, dXdr, dYdr, dZdr])
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
    print('*----- ARGS ------*')
    print(sys.argv)
    coords = str(sys.argv[4]).split('/')
    subdomain = {'longitude': slice(float(coords[0]), float(coords[1])),
                 'latitude': slice(float(coords[2]), float(coords[3]))}
    lcs = LCS(timestep=float(sys.argv[1]), timedim=str(sys.argv[2]), SETTLS_order=int(sys.argv[3]),
              subdomain=None)
    input_path = str(sys.argv[5])
    out = lcs(ds=input_path, isglobal=True, s=3e6)
    print('Saving to ' + str(sys.argv[6]))
    out.to_netcdf(sys.argv[6])
    subprocess.call(['rm', input_path])

    # ncpath = '/work/scratch-nopw/gmpp/experiment_timelen_8_902214ae-5f9a-45d8-b45c-d53239154b37/' \
    #          'input_partial_1981-01-01T00:00:00.000000000.nc'
    # subdomain = None
    # timestep = - 6 * 3600
    # SETTLS_order = 4
    # timedim='time'
    # import matplotlib.pyplot as plt
    # import cartopy.crs as ccrs
    # lcs = LCS(timestep=timestep, timedim=timedim, SETTLS_order=SETTLS_order, subdomain=subdomain)
    # out = lcs(ncpath, isglobal=True, s=3e6)
    # import cmasher as cmr
    # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Robinson()})
    # (.5*np.log(out)).isel(time=0).plot(ax=ax, vmin=0, vmax=2.5, transform=ccrs.PlateCarree(), cmap = cmr.freeze,
    #                        cbar_kwargs={'shrink': 0.6})
    # ax.coastlines(color='white')
    # plt.savefig('LagrangianCoherence/LCS/temp_figs.png', dpi=600)
    # plt.close()