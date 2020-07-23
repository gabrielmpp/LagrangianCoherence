import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import RectSphereBivariateSpline, CubicSpline, griddata
from copy import deepcopy


def parcel_propagation(U: xr.DataArray,
                       V: xr.DataArray,
                       timestep: float = 1,
                       propdim: str = "time",
                       verbose: bool = True,
                       s: float = None,
                       return_traj: bool = False,
                       SETTLS_order=0,
                       copy=False,
                       C=None,
                       Srcs=None):
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

    Lats must be [-90, 90]
    Lons must be [-180, 180]
    """
    if copy:
        U = U.copy()
        V = V.copy()

    tracer_account = False

    verboseprint = print if verbose else lambda *a, **k: None
    latmin = deepcopy(U.latitude.min().values) - np.abs(U.latitude.diff('latitude').values[0]) # offset so that the interp works
    lonmin = deepcopy(U.longitude.min().values) - np.abs(U.longitude.diff('longitude').values[0])
    U = U.assign_coords(latitude=(U.latitude.values - latmin) * np.pi / 180,
                        longitude=(U.longitude.values - lonmin) * np.pi / 180)
    V = V.assign_coords(latitude=(V.latitude.values - latmin) * np.pi / 180,
                        longitude=(V.longitude.values - lonmin) * np.pi / 180)


    U = U.sortby('longitude')
    V = V.sortby('longitude')
    U = U.sortby('latitude')
    V = V.sortby('latitude')

    if isinstance(C, type(None)):
        C = U.copy(data=np.zeros(shape=U.shape))
    else:
        C = C.assign_coords(latitude=(C.latitude.values - latmin) * np.pi / 180,
                            longitude=(C.longitude.values - lonmin) * np.pi / 180)
        C = C.sortby('longitude')
        C = C.sortby('latitude')
        tracer_account = True
        rel_contribution_list = []

    if isinstance(Srcs, type(None)):
        Srcs = U.copy(data=np.zeros(shape=U.shape))
    else:
        Srcs = Srcs.assign_coords(latitude=(Srcs.latitude.values - latmin) * np.pi / 180,
                            longitude=(Srcs.longitude.values - lonmin) * np.pi / 180)
        Srcs = Srcs.sortby('longitude')
        Srcs = Srcs.sortby('latitude')
        tracer_account = True
        rel_contribution_list = []

    earth_r = 6371000
    conversion_y = 1 / earth_r  # converting m/s to rad/s
    conversion_x = 1 / (earth_r * xr.apply_ufunc(lambda x: np.abs(np.cos(x)), U.latitude + latmin * np.pi/180))
    conversion_x, _ = xr.broadcast(conversion_x, U.isel({propdim: 0}))
    times = U[propdim].values.tolist()

    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating
    y_min = U.latitude.values.min()
    y_max = U.latitude.values.max()
    x_min = U.longitude.values.min()
    x_max = U.longitude.values.max()

    positions_y, positions_x = np.meshgrid(U.latitude.values, U.longitude.values)
    positions_x_t0 = positions_x.copy()
    positions_y_t0 = positions_y.copy()

    # positions_y
    pos_list_x = []
    pos_list_y = []
    # pos_list_x.append(positions_x)  # appending t=0
    # pos_list_y.append(positions_y)

    for time in times:
        verboseprint(f'Propagating time {time}')
        v_data = V.sel({propdim: time}).values

        interpolator_y = RectSphereBivariateSpline(V.latitude.values, V.longitude.values, v_data, s=s)
        va = interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)
        # va = griddata((positions_y.ravel(), positions_x.ravel()), v_data.ravel(),
        #               (positions_y.ravel(), positions_x.ravel()), method=method, rescale=True)
        # va = va.reshape(positions_y.shape)


        u_data = U.sel({propdim: time}).values
        interpolator_x = RectSphereBivariateSpline(U.latitude.values, U.longitude.values, u_data, s=s)
        ua = interpolator_x.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

        positions_y = positions_y + timestep * conversion_y * va
        positions_x = positions_x + timestep * conversion_x.values.T * ua
        # Hard boundary
        positions_y[np.where(positions_y < y_min)] = y_min
        positions_y[np.where(positions_y > y_max)] = y_max

        # Hard boundary
        positions_x[np.where(positions_x < x_min)] = x_min
        positions_x[np.where(positions_x > x_max)] = x_max

        # ECMWF's SETTLS algorithm for second-order advection accuracy (Hortal; 2002)
        k = 0
        while k < SETTLS_order:
            verboseprint(f'SETTLS iteration {k}')

            # ---- propagating positions ---- #
            v_t_depts = V.sel({propdim: time}).values
            interpolator = RectSphereBivariateSpline(V.latitude.values, V.longitude.values, v_t_depts, s=s)
            v_t_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            v_tprevious_depts = V.sel({propdim: time + timestep}, method='nearest').values
            interpolator = RectSphereBivariateSpline(V.latitude.values, V.longitude.values, v_tprevious_depts, s=s)
            v_tprevious_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            u_t_depts = U.sel({propdim: time}).values
            interpolator = RectSphereBivariateSpline(U.latitude.values, U.longitude.values, u_t_depts, s=s)
            u_t_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

            u_tprevious_depts = U.sel({propdim: time + timestep}, method='nearest').values
            interpolator = RectSphereBivariateSpline(U.latitude.values, U.longitude.values, u_tprevious_depts, s=s)
            u_tprevious_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            positions_y = positions_y + 0.5 * timestep * conversion_y * (va + 2*v_t_depts - v_tprevious_depts)

            positions_x = positions_x + 0.5 * timestep * conversion_x.values.T * (ua + 2*u_t_depts - u_tprevious_depts)

            # Hard boundary
            positions_y[np.where(positions_y < y_min)] = y_min
            positions_y[np.where(positions_y > y_max)] = y_max

            # Hard boundary
            positions_x[np.where(positions_x < x_min)] = x_min
            positions_x[np.where(positions_x > x_max)] = x_max

            k += 1
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)

        if tracer_account:

            srcs = Srcs.sel({propdim: time}).interp(latitude=positions_y.ravel(), longitude=positions_x.ravel(),
                                                    method='linear')
            c = C.sel({propdim: time}).interp(latitude=positions_y.ravel(), longitude=positions_x.ravel(),
                                                    method='linear')
            srcs = srcs.sortby('longitude')
            srcs = srcs.sortby('latitude')
            c = c.sortby('longitude')
            c = c.sortby('latitude')
            srcs = 1 - np.exp(-srcs/c)
            rel_contribution_list.append(srcs)

    U = U.assign_coords(latitude= U.latitude.values * 180 / np.pi + latmin,
                            longitude = U.longitude.values * 180 / np.pi + lonmin)
    V = V.assign_coords(latitude= V.latitude.values * 180 / np.pi + latmin,
                        longitude = V.longitude.values * 180 / np.pi + lonmin)
    for i in range(len(pos_list_x)):
        pos_list_x[i] = pos_list_x[i] * 180 / np.pi + lonmin
        pos_list_y[i] = pos_list_y[i] * 180 / np.pi + latmin
        pos_list_x[i] = xr.DataArray(pos_list_x[i].T, dims=['latitude', 'longitude'],
                                     coords=[U.latitude.values.copy(), U.longitude.values.copy()])
        pos_list_y[i] = xr.DataArray(pos_list_y[i].T, dims=['latitude', 'longitude'],
                                     coords=[U.latitude.values.copy(), U.longitude.values.copy()])
    # time_list = U[propdim].values.tolist()
    # time_list.append(pd.Timestamp(pd.Timestamp(U[propdim].values[-1]) + pd.Timedelta(str(timestep)+'s')))
    positions_x = xr.concat(pos_list_x, dim=pd.Index(pd.to_datetime(times), name=propdim))
    positions_y = xr.concat(pos_list_y, dim=pd.Index(pd.to_datetime(times), name=propdim))
    if not return_traj:
        positions_x = positions_x.isel({propdim: -1})
        positions_y = positions_y.isel({propdim: -1})

    return positions_x, positions_y


def parcel_propagation3D(U: xr.DataArray,
                         V: xr.DataArray,
                         W: xr.DataArray,
                         timestep: int = 1,
                         propdim: str = "time",
                         verbose: bool = True,
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

        for subtime in subtimes: # TODO: REMOVE this increase diffusion, see ECMWF lecture
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
    import matplotlib.pyplot as plt
    from xr_tools.tools import latlonsel

    data_dir = '/media/gab/gab_hd/data/'
    sel_slice = dict(
        lat=slice(-70, 15),
        lon=slice(-140, 20)

    )
    level=850
    W = xr.open_dataarray(data_dir + 'w_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 6))
    U = xr.open_dataarray(data_dir + 'u_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None,6))
    V = xr.open_dataarray(data_dir + 'v_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 6))
    U = U.assign_coords(longitude=(U['longitude'].values + 180) % 360 - 180)
    V = V.assign_coords(longitude=(V['longitude'].values + 180) % 360 - 180)
    U = latlonsel(U, **sel_slice, latname='latitude', lonname='longitude')
    V = latlonsel(V, **sel_slice, latname='latitude', lonname='longitude')
    x, y = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6*3600, SETTLS_order=4, return_traj=False)
    x2, y2 = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6*3600, SETTLS_order=0, return_traj=False)

    U = U.sortby('longitude')
    U = U.sortby('latitude')
    V = V.sortby('longitude')
    V = V.sortby('latitude')
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=[20, 10])
    # x.plot.contourf(transform=ccrs.PlateCarree(), levels=15, vmin=-140, vmax=20, cmap='nipy_spectral', ax=ax)
    p = ax.contourf(U.longitude.values, U.latitude.values, y2.values, levels=15, vmin=-70, vmax=15, cmap='RdBu')
    ax.streamplot(U.longitude.values, U.latitude.values, U.sel(time=x.time.values, level=level).values,
                  V.sel(time=x.time.values,level=level).values, color='black')
    ax.coastlines()
    plt.colorbar(p, ax=ax)
    plt.tight_layout()
    plt.show()

    (x-x2).plot(vmax=1e-1, vmin=-1e-1, cmap='RdBu'); plt.show()
    # x, y, z = parcel_propagation3D(U, V, W, timestep=6*3600, return_traj=True)
    print(x)