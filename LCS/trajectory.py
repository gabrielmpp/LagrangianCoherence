import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import RectSphereBivariateSpline, CubicSpline


def parcel_propagation(U: xr.DataArray,
                       V: xr.DataArray,
                       timestep: int = 1,
                       propdim: str = "time",
                       verbose: bool = True,
                       subtimes_len: int = 10,
                       s: float = 1e5,
                       return_traj: bool = False):
    """
    Method to propagate the parcel given u and v and, optionally, w (m/s)

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


    U.latitude.values = (90 + U.latitude.values) * np.pi / 180
    U.longitude.values = (180 + U.longitude.values) * np.pi / 180
    V.latitude.values = (90 + V.latitude.values) * np.pi / 180
    V.longitude.values = (180 + V.longitude.values) * np.pi / 180
    U = U.sortby('longitude')
    V = V.sortby('longitude')
    U = U.sortby('latitude')
    V = V.sortby('latitude')

    earth_r = 6371000
    conversion_y = 1 / earth_r  # converting m/s to rad/s
    conversion_x = 1 / (earth_r * xr.apply_ufunc(lambda x: np.abs(np.cos(x - 0.5 * np.pi)), U.latitude))
    conversion_x, _ = xr.broadcast(conversion_x, U.isel({propdim: 0}))
    times = U[propdim].values.tolist()
    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating

    positions_y, positions_x = np.meshgrid(U.latitude.values, U.longitude.values)
    # positions_y

    initial_pos = xr.DataArray()
    pos_list_x = []
    pos_list_y = []
    pos_list_x.append(positions_x)  # appending t=0
    pos_list_y.append(positions_y)
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
            interpolator_y = RectSphereBivariateSpline(V.latitude.values, V.longitude.values, v_data, s=s)

            positions_y = positions_y + \
                          subtimestep * conversion_y * \
                          interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            # Hard boundary
            positions_y[np.where(positions_y < 0)] = 0
            positions_y[np.where(positions_y > np.pi)] = np.pi

            u_data = U.sel({propdim: time}).values
            interpolator_x = RectSphereBivariateSpline(U.latitude.values, U.longitude.values, u_data)
            positions_x = positions_x + \
                          subtimestep * conversion_x.values.T * \
                          interpolator_x.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

            # Hard boundary
            positions_x[np.where(positions_x < 0)] = 0
            positions_x[np.where(positions_x > 2 * np.pi)] = 2 * np.pi
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)


    U.latitude.values = U.latitude.values * 180 / np.pi - 90
    U.longitude.values = U.longitude.values * 180 / np.pi - 180
    V.latitude.values = V.latitude.values * 180 / np.pi - 90
    V.longitude.values = V.longitude.values * 180 / np.pi - 180

    for i in range(len(pos_list_x)):
        pos_list_x[i] = pos_list_x[i] * 180 / np.pi - 180
        pos_list_y[i] = pos_list_y[i] * 180 / np.pi - 90
        pos_list_x[i] = xr.DataArray(pos_list_x[i].T, dims=['latitude', 'longitude'],
                               coords=[U.latitude.values.copy(), U.longitude.values.copy()])
        pos_list_y[i] = xr.DataArray(pos_list_y[i].T, dims=['latitude', 'longitude'],
                               coords=[U.latitude.values.copy(), U.longitude.values.copy()])
    if return_traj:
        time_list = [pd.Timestamp(x) for x in U[propdim].values]
        time_list.append(pd.Timestamp(pd.Timestamp(U[propdim].values[-1]) + pd.Timedelta(str(timestep)+'s')))
        positions_x = xr.concat(pos_list_x, dim=pd.Index(time_list, name=propdim))
        positions_y = xr.concat(pos_list_y, dim=pd.Index(time_list, name=propdim))
    else:
        positions_x = pos_list_x[-1]
        positions_y = positions_y[-1]

    return positions_x, positions_y


def parcel_propagation3D(U: xr.DataArray,
                         V: xr.DataArray,
                         W: xr.DataArray,
                         timestep: int = 1,
                         propdim: str = "time",
                         verbose: bool = True,
                         subtimes_len: int = 10,
                         s: float = 1e5,
                         return_traj: bool = False):
    """
    Method to propagate the parcel given u and v and, optionally, w (m/s)

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

    ds = xr.Dataset({'U': U, 'V': V, 'W': W})


    ds.latitude.values = (90 + ds.latitude.values) * np.pi / 180
    ds.longitude.values = (180 + ds.longitude.values) * np.pi / 180

    ds = ds.sortby('longitude')
    ds = ds.sortby('latitude')
    U, V, W = ds.U, ds.V, ds.W
    del ds

    earth_r = 6371000
    conversion_z = 1 / 100 # Pa to hPa
    conversion_y = 1 / earth_r  # converting m/s to rad/s
    conversion_x = 1 / (earth_r * xr.apply_ufunc(lambda x: np.abs(np.cos(x - 0.5 * np.pi)), U.latitude))
    conversion_x, _ = xr.broadcast(conversion_x, U.isel({propdim: 0}))
    times = U[propdim].values.tolist()
    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating

    positions_y, positions_x = np.meshgrid(U.latitude.values.copy(), U.longitude.values.copy())
    positions_z = W.level.values.copy()

    initial_pos = xr.DataArray()
    pos_list_x = []
    pos_list_y = []
    pos_list_z = []
    pos_list_x.append(positions_x)  # appending t=0
    pos_list_y.append(positions_y)
    pos_list_z.append(positions_z)
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
            # TODO: now v_data has a vertical dimension --- design new interpolation
            from scipy.interpolate import interpn
            interpn
            interpolator_y = interpn(V.latitude.values, V.longitude.values, v_data, s=s)

            positions_y = positions_y + \
                          subtimestep * conversion_y * \
                          interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            # Hard boundary
            positions_y[np.where(positions_y < 0)] = 0
            positions_y[np.where(positions_y > np.pi)] = np.pi

            u_data = U.sel({propdim: time}).values
            interpolator_x = RectSphereBivariateSpline(U.latitude.values, U.longitude.values, u_data)
            positions_x = positions_x + \
                          subtimestep * conversion_x.values.T * \
                          interpolator_x.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

            w_data = W.sel({propdim: time}).values
            interpolator_z = CubicSpline(W.level.values, w_data)
            positions_z = positions_z + \
                          subtimestep * conversion_z * \
                          interpolator_z(positions_z.ravel()).reshape(positions_z.shape)
            # Hard boundary
            positions_x[np.where(positions_x < 0)] = 0
            positions_x[np.where(positions_x > 2 * np.pi)] = 2 * np.pi
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)
        pos_list_z.append(positions_z)

    ds = xr.Dataset({'U': U, 'V': V, 'W': W})

    ds.latitude.values = ds.latitude.values * 180 / np.pi - 90
    ds.longitude.values = ds.longitude.values * 180 / np.pi - 180

    for i in range(len(pos_list_x)):
        pos_list_x[i] = pos_list_x[i] * 180 / np.pi - 180
        pos_list_y[i] = pos_list_y[i] * 180 / np.pi - 90
        pos_list_x[i] = xr.DataArray(pos_list_x[i].T, dims=['latitude', 'longitude'],
                               coords=[ds.latitude.values.copy(), ds.longitude.values.copy()])
        pos_list_y[i] = xr.DataArray(pos_list_y[i].T, dims=['latitude', 'longitude'],
                               coords=[ds.latitude.values.copy(), ds.longitude.values.copy()])
        pos_list_z[i] = xr.DataArray(pos_list_z[i].T, dims=['latitude', 'longitude'],
                               coords=[ds.latitude.values.copy(), ds.longitude.values.copy()])
    if return_traj:
        time_list = [pd.Timestamp(x) for x in ds[propdim].values]
        time_list.append(pd.Timestamp(pd.Timestamp(U[propdim].values[-1]) + pd.Timedelta(str(timestep)+'s')))
        positions_x = xr.concat(pos_list_x, dim=pd.Index(time_list, name=propdim))
        positions_y = xr.concat(pos_list_y, dim=pd.Index(time_list, name=propdim))
        positions_z = xr.concat(pos_list_z, dim=pd.Index(time_list, name=propdim))

    else:
        positions_x = pos_list_x[-1]
        positions_y = pos_list_y[-1]
        positions_z = pos_list_z[-1]

    return positions_x, positions_y, positions_z


if __name__ == '__main__':
    W = xr.open_dataarray('../../../data/w_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 4))
    U = xr.open_dataarray('../../../data/u_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 4))
    V = xr.open_dataarray('../../../data/v_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 4))
    x, y, z = parcel_propagation3D(U, V, W, timestep=6*3600, return_traj=True)
    print(x)