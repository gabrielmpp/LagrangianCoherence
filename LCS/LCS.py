import xarray as xr
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from numba import jit
import numba

LCS_TYPES = ['attracting', 'repelling']


class LCS:
    """
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    """
    earth_r = 6371000

    def __init__(self, lcs_type: str, timestep: float = 1, dataarray_template=None, timedim='time',
                 shearless=False, subtimes_len=1):
        """

        :param lcs_type:
        :param timestep:
        :param dataarray_template:
        :param timedim:
        :param shearless:
        :param subtimes_len:
        """
        self.subtimes_len = subtimes_len
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.timedim = timedim
        self.shearless = shearless

        self.dataarray_template = dataarray_template

    def __call__(self, ds: xr.Dataset = None, u: xr.DataArray = None, v: xr.DataArray = None,
                 verbose=True) -> xr.DataArray:

        """
        :param ds: xarray dataset containing u and v as variables
        :param u: xarray datarray containing u-wind component
        :param v: xarray dataarray containing v-wind component
        :return: xarray dataarray of eigenvalue
        """
        global verboseprint

        verboseprint = print if verbose else lambda *a, **k: None

        timestep = self.timestep
        timedim = self.timedim
        self.verbose = verbose
        if isinstance(ds, xr.Dataset):
            u = ds.u
            v = ds.v
        if isinstance(self.dataarray_template, xr.DataArray):
            template = self.dataarray_template
            u = xr.DataArray(u, coords=template.coords, dims=template.dims)
            v = xr.DataArray(v, coords=template.coords, dims=template.dims)

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
        def_tensor = self._compute_deformation_tensor(u, v, timestep)
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

    def _compute_eigenvalues(self, def_tensor: np.array) -> np.array:
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)

        if self.lcs_type == 'repelling':
            eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        elif self.lcs_type == 'attracting':
            eigenvalues = min(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        else:
            raise ValueError("lcs_type {lcs_type} not supported".format(lcs_type=self.lcs_type))

        eigenvalues = np.repeat(eigenvalues, 4).reshape(
            [4])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        return eigenvalues

    def _compute_deformation_tensor(self, u: xr.DataArray, v: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        x_departure, y_departure = parcel_propagation(u, v, timestep, propdim=self.timedim, subtimes_len=self.subtimes_len)
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
        if self.shearless:
            dydx = dydx * 0
            dxdy = dxdy * 0
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

    def _compute_deformation_tensor_old(self, u: xr.DataArray, v: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        u, v, eigengrid = interpolate_c_stagger(u, v)

        # ------- Computing dy_futur/dx -------- #
        # When derivating with respect to x we use u coords in the Arakawa C grid
        y_futur = u.y + v.interp(latitude=u.latitude, longitude=u.longitude,
                                 kwargs={'fill_value': None}) * timestep  # fill_value=None extrapolates
        dx = u.x.diff('longitude')
        dydx = y_futur.diff('longitude') / dx
        dydx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dx -------- #
        x_futur = u.x + u * timestep
        dx = u.x.diff('longitude')
        dxdx = x_futur.diff('longitude') / dx
        dxdx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dy -------- #
        # When derivating with respect to y we use v coords in the Arakawa C grid
        x_futur = v.x + u.interp(latitude=v.latitude, longitude=v.longitude,
                                 kwargs={'fill_value': None}) * timestep  # fill_value=None extrapolates
        dy = v.y.diff('latitude')
        dxdy = x_futur.diff('latitude') / dy
        dxdy['latitude'] = eigengrid.latitude

        # ------- Computing dy_futur/dy -------- #
        y_futur = v.y + v * timestep
        dy = v.y.diff('latitude')
        dydy = y_futur.diff('latitude') / dy
        dydy['latitude'] = eigengrid.latitude

        dxdx = dxdx.transpose('latitude', 'longitude').drop('x').drop('y')
        dxdy = dxdy.transpose('latitude', 'longitude').drop('x') * 0
        dydy = dydy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydx = dydx.transpose('latitude', 'longitude').drop('y') * 0
        dxdx.name = 'dxdx'
        dxdy.name = 'dxdy'
        dydy.name = 'dydy'
        dydx.name = 'dydx'

        def_tensor = xr.merge([dxdx, dxdy, dydx, dydy])
        def_tensor = def_tensor.to_array()
        def_tensor = def_tensor.rename({'variable': 'derivatives'})
        def_tensor = def_tensor.transpose('derivatives', 'latitude', 'longitude')
        return def_tensor


def parcel_propagation(u, v, timestep, propdim="time", verbose=True, subtimes_len=10):
    import scipy.interpolate as interp
    """
    Method to propagate the parcel given u and v

    :param propdim: str, dimension name for time
    :param timestep: double, timestep in seconds
    :param u: xr.DataArray zonal wind
    :param v: xr.DataArray meridional wind
    :return: tuple of xr.DataArrays containing the final positions of the trajectories
    """
    verboseprint = print if verbose else lambda *a, **k: None
    earth_r = 6371000
    conversion_y = 180 / (np.pi * earth_r)  # converting m/s to rad/s
    conversion_x = 180 * ((earth_r * np.pi) ** (-1)) * xr.apply_ufunc(lambda x: np.cos(x * np.pi / 180), u.latitude) ** (-1)
    conversion_x, _ = xr.broadcast(conversion_x, u.isel({propdim: 0}))
    times = u[propdim].values.tolist()
    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating

    positions_x, positions_y = np.meshgrid(u.longitude.values, u.latitude.values)

    initial_pos = xr.DataArray()

    for time in times:
        verboseprint(f'Propagating time {time}')
        subtimes = np.arange(0, subtimes_len, 1).tolist()

        for subtime in subtimes:
            verboseprint(f'Propagating subtime {subtime}')

            subtimestep = timestep / subtimes_len

            lat = positions_y[:, 0]  # lat is constant along cols
            lon = positions_x[0, :]  # lon is constant along rows

            # ---- propagating positions ---- #

            positions_y = positions_y + \
                          subtimestep * conversion_y * \
                          v.sel({propdim: time}).interp(latitude=lat.tolist(), method='linear',
                                                        longitude=lon.tolist(),
                                                        kwargs={'fill_value': None}).values

            positions_x = positions_x + \
                          subtimestep * conversion_x.values * \
                          u.sel({propdim: time}).interp(latitude=lat.tolist(), method='linear',
                                                        longitude=lon.tolist(),
                                                        kwargs={'fill_value': None}).values

    positions_x = xr.DataArray(positions_x, dims=['latitude', 'longitude'],
                               coords=[u.latitude.values, u.longitude.values])
    positions_y = xr.DataArray(positions_y, dims=['latitude', 'longitude'],
                               coords=[u.latitude.values, u.longitude.values])

    return positions_x, positions_y


def find_maxima(eigenarray: xr.DataArray):
    data = eigenarray.values
    neighborhood_size = 4
    threshold = 0.08

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(diff)
    diff_array = xr.DataArray(diff, coords=eigenarray.coords, dims=eigenarray.dims)
    out_array = xr.DataArray(data_max, coords=eigenarray.coords, dims=eigenarray.dims)
    labeled_array = xr.DataArray(labeled, coords=eigenarray.coords, dims=eigenarray.dims)
    return out_array, diff_array, labeled_array


@jit(parallel=True)
def compute_eigenvalues(arrays_list):
    out_list = []
    for i in numba.prange(len(arrays_list)):
        def_tensor = arrays_list[i]
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)
        eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        eigenvalues = np.repeat(eigenvalues, 4).reshape([4])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    lat1 = -80
    lat2 = 20
    lon1 = -80
    lon2 = -30
    dx = 0.5
    earth_r = 6371000
    subtimes_len = 2
    timestep = 6 * 3600
    ntime = 4
    ky = 10
    kx = 40
    time_dir = 'backward'
    timestep = -timestep if time_dir == 'backward' else timestep

    nlat = int((lat2 - lat1) / dx)
    nlon = int((lon2 - lon1) / dx)
    latitude = np.linspace(lat1, lat2, nlat)
    longitude = np.linspace(lon1, lon2, nlon)
    time = pd.date_range("2000-01-01T00:00:00", periods=ntime, freq="6H")
    time_idx = np.array([x for x in range(len(time))])
    frq = 0.25
    u_data = 20 * np.ones([nlat, nlon, len(time)]) * (np.sin(ky*np.pi*latitude/180)**2).reshape([nlat,1,1]) * np.cos(time_idx * frq).reshape([1,1,len(time)])
    v_data = 40 * np.ones([nlat, nlon, len(time)]) * (np.sin(kx*np.pi*longitude/360)**2).reshape([1,nlon,1]) * np.cos(time_idx * frq).reshape([1,1,len(time)])

    u = xr.DataArray(u_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude, 'longitude': longitude, 'time': time})
    v = xr.DataArray(v_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude, 'longitude': longitude, 'time': time})
    lcs = LCS(lcs_type='repelling', timestep=timestep, timedim='time', subtimes_len=subtimes_len)

    u.name = 'u'
    v.name = 'v'
    ds = xr.merge([u, v])
    ftle = lcs(ds)
    dep_x, dep_y = parcel_propagation(u, v, timestep=timestep, subtimes_len=subtimes_len)
    origin = np.meshgrid(longitude, latitude)[0]
    origin.shape
    displacement = dep_x.copy(data=dep_x - origin)
    plt.streamplot(longitude, latitude, u.isel(time=0).values, v.isel(time=0).values)
    (displacement / len(time)).plot()
    plt.show()
    dep_x.plot(vmax=-30, vmin=-80,cmap='rainbow')
    plt.show()
    dep_y.plot(vmax=-6, vmin=-60,cmap='rainbow')
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

    ftle.isel(time=0).plot()
    plt.show()
    print("s")