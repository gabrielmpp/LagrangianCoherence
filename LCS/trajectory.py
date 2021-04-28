import xarray as xr
import numpy as np
import pandas as pd
from LagrangianCoherence.LCS.tools import xr_map_coordinates
import cftime


def parcel_propagation(U: xr.DataArray,
                       V: xr.DataArray,
                       timestep: float = 1,
                       propdim: str = "time",
                       verbose: bool = True,
                       return_traj: bool = False,
                       SETTLS_order=0,
                       copy=False,
                       interp_order=3,
                       cyclic_xboundary=False,
                       ):
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

    U = U.sortby('longitude')
    V = V.sortby('longitude')
    U = U.sortby('latitude')
    V = V.sortby('latitude')

    earth_r = 6371000
    conversion_y = 180 / (earth_r * np.pi)  # converting m/s to deg/s
    conversion_x = 180 / (np.pi * earth_r * xr.apply_ufunc(lambda x: np.abs(np.cos(x * np.pi / 180)), U.latitude))
    conversion_x, _ = xr.broadcast(conversion_x, U.isel({propdim: 0}))
    times = U[propdim].values.tolist()
    if timestep < 0:
        times.reverse()  # inplace

    # initializing and integrating
    y_min = U.latitude.values.min()
    y_max = U.latitude.values.max()
    x_min = U.longitude.values.min()
    x_max = U.longitude.values.max()

    positions_y_t0 = U.latitude.values
    positions_x_t0 = U.longitude.values
    positions_x, positions_y = np.meshgrid(positions_x_t0, positions_y_t0)
    # positions_y
    pos_list_x = []
    pos_list_y = []
    # pos_list_x.append(positions_x)  # appending t=0
    # pos_list_y.append(positions_y)
    pos_list_x.append(positions_x)
    pos_list_y.append(positions_y)


    for time_idx, time in enumerate(times[:-1]):
        verboseprint(f'Propagating time {time}')
        va = xr_map_coordinates(V.isel({propdim: time_idx}), positions_x, positions_y, order=interp_order)

        ua = xr_map_coordinates(U.isel({propdim: time_idx}), positions_x, positions_y, order=interp_order)

        positions_y = positions_y + timestep * conversion_y * va
        positions_x = positions_x + timestep * conversion_x * ua
        # Hard y boundary
        positions_y = positions_y.where(positions_y > y_min, y_min)
        positions_y = positions_y.where(positions_y < y_max, y_max)

        if cyclic_xboundary:
            positions_x = positions_x.where(positions_x > -180, positions_x % 180)
            positions_x = positions_x.where(positions_x < 180, -180 + (positions_x % 180))
        else:
            positions_x[np.where(positions_x < x_min)] = x_min
            positions_x[np.where(positions_x > x_max)] = x_max
        # ECMWF's SETTLS algorithm for second-order advection accuracy (Hortal; 2002)
        k = 0
        while k < SETTLS_order:
            verboseprint(f'SETTLS iteration {k}')

            # ---- propagating positions ---- #

            v_t_depts = xr_map_coordinates(V.isel({propdim: time_idx}), positions_x, positions_y, order=interp_order)
            v_tprevious_depts = xr_map_coordinates(V.isel({propdim: time_idx + 1}), positions_x, positions_y, order=interp_order)
            u_t_depts = xr_map_coordinates(U.isel({propdim: time_idx}), positions_x, positions_y, order=interp_order)
            u_tprevious_depts = xr_map_coordinates(U.isel({propdim: time_idx + 1}), positions_x, positions_y, order=interp_order)

            positions_y = positions_y + 0.5 * timestep * conversion_y * (va + 2 * v_t_depts - v_tprevious_depts)

            positions_x = positions_x + 0.5 * timestep * conversion_x * (ua + 2 * u_t_depts - u_tprevious_depts)

            # Hard boundary
            positions_y = positions_y.where(positions_y > y_min, y_min)
            positions_y = positions_y.where(positions_y < y_max, y_max)

            if cyclic_xboundary:
                positions_x = positions_x.where(positions_x > -180, positions_x % 180)
                positions_x = positions_x.where(positions_x < 180, -180 + (positions_x % 180))
            else:
                positions_x[np.where(positions_x < x_min)] = x_min
                positions_x[np.where(positions_x > x_max)] = x_max
            k += 1
        pos_list_x.append(positions_x)
        pos_list_y.append(positions_y)

    if return_traj:
        assert not isinstance(times[0], cftime.Datetime360Day), \
            'Cannot return trajectories with time cooridnates cftime.Datetime360Day.'
        times = [times[0] - timestep] + times
        positions_x = xr.concat(pos_list_x, dim=pd.Index(pd.to_datetime(times), name=propdim))
        positions_y = xr.concat(pos_list_y, dim=pd.Index(pd.to_datetime(times), name=propdim))
    else:
        positions_x = pos_list_x[-1].assign_coords({propdim: times[-1]})  # .expand_dims(propdim)
        positions_y = pos_list_y[-1].assign_coords({propdim: times[-1]})  # .expand_dims(propdim)

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
    level = 850
    W = xr.open_dataarray(data_dir + 'w_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 8))
    U = xr.open_dataarray(data_dir + 'u_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 8))
    V = xr.open_dataarray(data_dir + 'v_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 8))
    U = U.assign_coords(longitude=(U['longitude'].values + 180) % 360 - 180)
    V = V.assign_coords(longitude=(V['longitude'].values + 180) % 360 - 180)
    lats = np.linspace(-89.5, 89.5, 180 * 2)
    lons = np.linspace(-179.5, 179.5, 360 * 2)
    U = U.interp(latitude=lats, longitude=lons, method='linear')
    V = V.interp(latitude=lats, longitude=lons, method='linear')
    # U = latlonsel(U, **sel_slice, latname='latitude', lonname='longitude')
    # V = latlonsel(V, **sel_slice, latname='latitude', lonname='longitude')
    x, y = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6 * 3600,
                              SETTLS_order=4, return_traj=True,
                              s=U.isel(time=0, level=0).size * U.isel(time=0, level=0).std(), cyclic_xboundary=True,
                              pole_continuity=False)
    x1, y1 = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6 * 3600,
                                SETTLS_order=4, return_traj=True,
                                s=U.isel(time=0, level=0).size * U.isel(time=0, level=0).std(), cyclic_xboundary=False,
                                pole_continuity=False)
    x2, y2 = parcel_propagation(U.sel(level=level), V.sel(level=level), timestep=-6 * 3600,
                                SETTLS_order=0, return_traj=False)

    for i in range(x.time.values.shape[0]):
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Orthographic(-50, -30)})
        p = (y - y1).isel(time=i).plot(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap='RdBu',
                                       add_colorbar=False)
        (x - x1).isel(time=i).plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap='RdBu',
                                   add_colorbar=False)
        axs[0].coastlines()
        axs[1].coastlines()
        axs[0].set_title('Lat error')
        axs[1].set_title('Lon error')
        plt.colorbar(p, ax=axs, orientation='horizontal')
        plt.show()
    for i in range(x.time.values.shape[0]):
        fig, axs = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        p = x.isel(time=i).sel(latitude=slice(-40, 40)).plot(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-90 * 2,
                                                             vmax=90 * 2, cmap=cmr.fusion,
                                                             add_colorbar=False)
        y.isel(time=i).sel(latitude=slice(-40, 40)).plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-90, vmax=90,
                                                         cmap=cmr.seasons,
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
                  V.sel(time=x.time.values, level=level).values, color='black')
    ax.coastlines()
    plt.colorbar(p, ax=ax)
    plt.tight_layout()
    plt.show()

    (x - x2).plot(vmax=1e-1, vmin=-1e-1, cmap='RdBu');
    plt.show()
    # x, y, z = parcel_propagation3D(U, V, W, timestep=6*3600, return_traj=True)
    print(x)
