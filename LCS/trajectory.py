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
                       Srcs=None,
                       s_is_error=False,
                       cyclic_xboundary=False,
                       pole_continuity=True):
    """
    Lagrangian 2-time-level advection scheme

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
        or data error used for estimating the smoothing factor internally
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
    latmin=-90
    lonmin=-180
    U = U.assign_coords(latitude=(U.latitude.values - latmin) * np.pi / 180,
                        longitude=(U.longitude.values - lonmin) * np.pi / 180)
    V = V.assign_coords(latitude=(V.latitude.values - latmin) * np.pi / 180,
                        longitude=(V.longitude.values - lonmin) * np.pi / 180)
    if s_is_error:
        s = (s ** 2) * U.isel({propdim: 0}).size ** 2

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
        tracer_list = []

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
        times = [times[0] - timestep] + times
    else:
        # times = times + [times[-1] + timestep]
        times = [times[0] - timestep] + times
    # initializing and integrating
    y_min = U.latitude.values.min()
    y_max = U.latitude.values.max()
    x_min = U.longitude.values.min()
    x_max = U.longitude.values.max()

    positions_y_t0 = U.latitude.values
    positions_x_t0 = U.longitude.values
    positions_y, positions_x = np.meshgrid(positions_y_t0, positions_x_t0)
    # positions_y
    pos_list_x = []
    pos_list_y = []
    # pos_list_x.append(positions_x)  # appending t=0
    # pos_list_y.append(positions_y)
    pos_list_x.append(positions_x)
    pos_list_y.append(positions_y)
    if tracer_account:
        tracer_list.append(C.sel({propdim: times[0]}, method='nearest').values.T) #  TODO WARNING: temporarily using nearest
    for time in times[1:]:
        verboseprint(f'Propagating time {time}')
        v_data = V.sel({propdim: time}).values

        interpolator_y = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, v_data, s=s, pole_continuity=pole_continuity)
        va = interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)
        # va = griddata((positions_y.ravel(), positions_x.ravel()), v_data.ravel(),
        #               (positions_y.ravel(), positions_x.ravel()), method=method, rescale=True)
        # va = va.reshape(positions_y.shape)


        u_data = U.sel({propdim: time}).values
        interpolator_x = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, u_data, s=s, pole_continuity=pole_continuity)
        ua = interpolator_x.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

        positions_y = positions_y + timestep * conversion_y * va
        positions_x = positions_x + timestep * conversion_x.values.T * ua
        # Hard y boundary
        positions_y[np.where(positions_y < y_min)] = y_min
        positions_y[np.where(positions_y > y_max)] = y_max

        if cyclic_xboundary:
            positions_x[np.where(positions_x < 0)] = positions_x[np.where(positions_x < 0)] % (2 * np.pi)
            positions_x[np.where(positions_x > 2 * np.pi)] = positions_x[np.where(positions_x > 2 * np.pi)] % (2 * np.pi)
        else:
            positions_x[np.where(positions_x < x_min)] = x_min
            positions_x[np.where(positions_x > x_max)] =  x_max
        # ECMWF's SETTLS algorithm for second-order advection accuracy (Hortal; 2002)
        k = 0
        while k < SETTLS_order:
            verboseprint(f'SETTLS iteration {k}')

            # ---- propagating positions ---- #
            v_t_depts = V.sel({propdim: time}).values
            interpolator = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, v_t_depts, s=s, pole_continuity=pole_continuity)
            v_t_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            v_tprevious_depts = V.sel({propdim: time + timestep}, method='nearest').values
            interpolator = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, v_tprevious_depts, s=s, pole_continuity=pole_continuity)
            v_tprevious_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            u_t_depts = U.sel({propdim: time}).values
            interpolator = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, u_t_depts, s=s, pole_continuity=pole_continuity)
            u_t_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)

            u_tprevious_depts = U.sel({propdim: time + timestep}, method='nearest').values
            interpolator = RectSphereBivariateSpline(positions_y_t0, positions_x_t0, u_tprevious_depts, s=s, pole_continuity=pole_continuity)
            u_tprevious_depts = interpolator.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

            positions_y = positions_y + 0.5 * timestep * conversion_y * (va + 2*v_t_depts - v_tprevious_depts)

            positions_x = positions_x + 0.5 * timestep * conversion_x.values.T * (ua + 2*u_t_depts - u_tprevious_depts)

            # Hard boundary
            positions_y[np.where(positions_y < y_min)] = y_min
            positions_y[np.where(positions_y > y_max)] = y_max

            if cyclic_xboundary:
                positions_x[np.where(positions_x < 0)] = positions_x[np.where(positions_x < 0)] % (2 * np.pi)
                positions_x[np.where(positions_x > 2 * np.pi)] = positions_x[np.where(positions_x > 2 * np.pi)] % (2 * np.pi)
            else:
                positions_x[np.where(positions_x < x_min)] = x_min
                positions_x[np.where(positions_x > x_max)] = x_max
            k += 1
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)

        if tracer_account:
            c_data = C.sel({propdim: time}, method='nearest').values  # TODO WARNING

            interpolator_c = RectSphereBivariateSpline(V.latitude.values, V.longitude.values, c_data, s=s)
            c = interpolator_c.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_x.shape)
            tracer_list.append(c)

    U = U.assign_coords(latitude= U.latitude.values * 180 / np.pi + latmin,
                            longitude = U.longitude.values * 180 / np.pi + lonmin)
    V = V.assign_coords(latitude= V.latitude.values * 180 / np.pi + latmin,
                        longitude = V.longitude.values * 180 / np.pi + lonmin)
    C = C.assign_coords(latitude= C.latitude.values * 180 / np.pi + latmin,
                        longitude = C.longitude.values * 180 / np.pi + lonmin)
    for i in range(len(pos_list_x)):
        pos_list_x[i] = pos_list_x[i] * 180 / np.pi + lonmin
        pos_list_y[i] = pos_list_y[i] * 180 / np.pi + latmin
        pos_list_x[i] = xr.DataArray(pos_list_x[i].T, dims=['latitude', 'longitude'],
                                     coords=[U.latitude.values.copy(), U.longitude.values.copy()])
        pos_list_y[i] = xr.DataArray(pos_list_y[i].T, dims=['latitude', 'longitude'],
                                     coords=[U.latitude.values.copy(), U.longitude.values.copy()])
        if tracer_account:
            print(tracer_list[i].shape)
            tracer_list[i] = xr.DataArray(tracer_list[i].T, dims=['latitude', 'longitude'],
                                         coords=[C.latitude.values.copy(), C.longitude.values.copy()])
    # time_list = U[propdim].values.tolist()
    # time_list.append(pd.Timestamp(pd.Timestamp(U[propdim].values[-1]) + pd.Timedelta(str(timestep)+'s')))
    positions_x = xr.concat(pos_list_x, dim=pd.Index(pd.to_datetime(times), name=propdim))
    positions_y = xr.concat(pos_list_y, dim=pd.Index(pd.to_datetime(times), name=propdim))
    if tracer_account:
        tracer = xr.concat(tracer_list, dim=pd.Index(pd.to_datetime(times), name=propdim))

    if not return_traj:
        positions_x = positions_x.isel({propdim: -1})
        positions_y = positions_y.isel({propdim: -1})
    if tracer_account:
        return positions_x, positions_y, tracer
    else:
        return positions_x, positions_y



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from xr_tools.tools import latlonsel
    import cartopy.crs as ccrs
    import cmasher as cmr
    data_dir = '/media/gab/gab_hd2/data/'
    sel_slice = dict(
        latitude=slice(-70, 15),
        longitude=slice(-140, 20)

    )
    level=850
    W = xr.open_dataarray(data_dir + 'w_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 8))
    U = xr.open_dataarray(data_dir + 'u_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None,8))
    V = xr.open_dataarray(data_dir + 'v_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 8))
    U = U.assign_coords(longitude=(U['longitude'].values + 180) % 360 - 180)
    V = V.assign_coords(longitude=(V['longitude'].values + 180) % 360 - 180)
    lats=np.linspace(-89.5, 89.5, 180*2)
    lons=np.linspace(-179.5, 179.5, 360*2)
    U = U.interp(latitude=lats, longitude=lons, method='linear')
    V = V.interp(latitude=lats, longitude=lons, method='linear')
    # U = latlonsel(U, **sel_slice, latname='latitude', lonname='longitude')
    # V = latlonsel(V, **sel_slice, latname='latitude', lonname='longitude')
    x, y = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6*3600,
                                  SETTLS_order=4, return_traj=True, s=U.isel(time=0,level=0).size  * U.isel(time=0,level=0).std(), cyclic_xboundary=True,
                              pole_continuity=False)
    x1, y1 = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6*3600,
                              SETTLS_order=4, return_traj=True, s=U.isel(time=0,level=0).size  * U.isel(time=0,level=0).std(), cyclic_xboundary=False, pole_continuity=False)
    x2, y2 = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6*3600,
                                SETTLS_order=0, return_traj=False)

    for i in range(x.time.values.shape[0]):
        fig, axs = plt.subplots(1,2, subplot_kw={'projection': ccrs.Orthographic(-50, -30)})
        p = (y-y1).isel(time=i).plot(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap='RdBu', add_colorbar=False)
        (x-x1).isel(time=i).plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap='RdBu', add_colorbar=False)
        axs[0].coastlines()
        axs[1].coastlines()
        axs[0].set_title('Lat error')
        axs[1].set_title('Lon error')
        plt.colorbar(p, ax=axs, orientation='horizontal')
        plt.show()
    for i in range(x.time.values.shape[0]):
        fig, axs = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        p = x.isel(time=i).sel(latitude=slice(-40, 40)).plot(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-90*2, vmax=90*2, cmap=cmr.fusion,
                                       add_colorbar=False)
        y.isel(time=i).sel(latitude=slice(-40, 40)).plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-90, vmax=90, cmap=cmr.seasons,
                                   add_colorbar=False)

        # p2 = y.isel(time=i).sel(latitude=slice(-80, 80)).plot(ax=axs[1,0], transform=ccrs.PlateCarree(), vmin=-90, vmax=90, cmap='RdBu',
        #                                add_colorbar=False)
        # y1.isel(time=i).sel(latitude=slice(-80, 80)).plot(ax=axs[1,1], transform=ccrs.PlateCarree(), vmin=-90, vmax=90, cmap='RdBu',
        #                            add_colorbar=False)
        for ax in axs.flatten():
            ax.coastlines()
        axs[0].set_title('Longitudinal mixing')
        axs[1].set_title('Latitudinal mixing')
        # axs[1, 0].set_title('Cyclic')
        # axs[1, 1].set_title('Solid')
        # plt.colorbar(p, ax=axs, orientation='horizontal')
        plt.savefig(f'temp_figs_{i:02d}_tropics.png', dpi=600, boundary='trim')
        plt.close()
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