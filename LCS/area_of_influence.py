import xarray as xr
import numpy as np
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from LagrangianCoherence.LCS.trajectory import parcel_propagation
from scipy.ndimage import gaussian_filter
import matplotlib
import time
import scipy.interpolate as interp
import cmasher as cmr
from skimage.morphology import skeletonize, binary_erosion, binary_dilation


def find_area(ftle, eigvectors, ridges, qsat=None, qdpt=None):
    """
    Funcion to identify areas where attracting LCSs influence rainfall by compressing advected trajectories in
    a regular lat-lon grid.
    Parameters
    ----------
    ftle
    ridges
    qsat
    qdpt

    Returns
    -------
    xr.DataArray

    """
    if qsat is None or qdpt is None:
        saturation_ratio = .5
    else:
        saturation_ratio = qdpt / qsat
    ftle = ftle.copy()
    eigvectors = eigvectors.copy()
    ridges = ridges.copy()
    ftle = ftle.sortby('latitude').sortby('longitude')
    eigvectors = eigvectors.sortby('latitude').sortby('longitude')
    ridges = ridges.sortby('latitude').sortby('longitude')
    # re = 6371  # km
    #
    # dx = np.abs(re * np.cos(ftle.latitude * np.pi/180) * ftle.longitude.diff('longitude'))
    # dy = np.abs(re * ftle.latitude.diff('latitude'))

    res = ftle['latitude'].diff('latitude').values[0]
    # res = 1  # rd is the unitary radius.
    sigma = np.exp(ftle.copy())  # should I consider time inverval length here?
    sigma = sigma * ridges
    normal_radius = sigma * saturation_ratio
    latitudes = sigma.latitude.values
    longitudes = sigma.longitude.values
    sigma = sigma.unstack()
    bounds = sigma.copy(data=np.zeros(sigma.shape))
    bounds = bounds.transpose('latitude', 'longitude')  # making sure of the order of the array
    sigma = sigma.stack(points=['latitude', 'longitude'])
    eigvectors = eigvectors.stack(points=['latitude', 'longitude'])
    normal_radius = normal_radius.stack(points=['latitude', 'longitude'])
    sigma = sigma.dropna('points')
    eigvectors = eigvectors.dropna('points')
    for pt in sigma.points.values:
        eigv_pt = eigvectors.sel(points=pt)
        norm_pt = normal_radius.sel(points=pt)
        x_upper = pt[1] + eigv_pt.isel(eigvectors=1) * norm_pt
        y_upper = pt[0] + eigv_pt.isel(eigvectors=0) * norm_pt
        x_lower = pt[1] - np.abs(eigv_pt.isel(eigvectors=1)) * norm_pt
        y_lower = pt[0] - np.abs(eigv_pt.isel(eigvectors=0)) * norm_pt
        xx = x_lower.copy()
        yy = y_lower.copy()
        D = 0
        while D <= 2 * norm_pt:
            xx += np.abs(eigv_pt.isel(eigvectors=1)) * res
            yy += np.abs(eigv_pt.isel(eigvectors=0)) * res
            xx_idx = np.argmin(np.abs(longitudes - xx.values))
            yy_idx = np.argmin(np.abs(latitudes - yy.values))
            bounds.values[yy_idx, xx_idx] = 1
            D = ((xx - x_lower) ** 2 + (yy - y_lower) ** 2) ** 0.5

        #
        # x_upper = np.argmin(np.abs(longitudes - x_upper.values))
        # x_lower = np.argmin(np.abs(longitudes - x_lower.values))
        # y_lower = np.argmin(np.abs(latitudes - y_lower.values))
        # y_upper = np.argmin(np.abs(latitudes - y_upper.values))

    return bounds


if __name__ == '__main__':

    # ---- Preparing input ---- #
    basepath = '/home/gab/phd/data/ERA5/'
    u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
    v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
    tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
    sfcpres_filepath = basepath + 'mslpres_ERA5_6hr_2020010100-2020123118.nc'
    orog_filepath = basepath + 'geopotential_orography.nc'
    data_dir = '/home/gab/phd/data/composites_cz/'
    pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'

    # timesel = sys.argv[1]
    timesel = slice('2020-01-20', '2020-02-28')
    subdomain = {'latitude': slice(-40, 15),
                 'longitude': slice(-90, -32)}
    u = xr.open_dataarray(u_filepath, chunks={'time': 140})
    u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
    u = u.sortby('longitude')
    u = u.sortby('latitude')

    u = u.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')
    u = u.sel(time=timesel)
    u = u.load()

    v = xr.open_dataarray(v_filepath, chunks={'time': 140})
    v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
    v = v.sortby('longitude')
    v = v.sortby('latitude')
    v = v.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')
    v = v.sel(time=timesel)
    v = v.load()

    tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
    tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
    tcwv = tcwv.sortby('longitude')
    tcwv = tcwv.sortby('latitude')
    tcwv = tcwv.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')
    tcwv = tcwv.sel(time=timesel)
    tcwv = tcwv.load()

    sfcpres = xr.open_dataarray(sfcpres_filepath, chunks={'time': 140})
    sfcpres = sfcpres.assign_coords(longitude=(sfcpres.coords['longitude'].values + 180) % 360 - 180)
    sfcpres = sfcpres.sortby('longitude')
    sfcpres = sfcpres.sortby('latitude')
    sfcpres = sfcpres.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')
    sfcpres = sfcpres.sel(time=timesel)
    sfcpres = sfcpres.load()
    mslpres = sfcpres
    # orog = xr.open_dataarray(orog_filepath, chunks={'time': 140})
    # orog = orog.assign_coords(longitude=(orog.coords['longitude'].values + 180) % 360 - 180)
    # orog = orog.sortby('longitude')
    # orog = orog.sortby('latitude')
    # orog = orog.sel(latitude=slice(-75, 60), longitude=slice(-150, 45))
    # orog = orog.load()
    # orog = orog.isel(time=0).drop('time')
    # orog = orog / 9.8
    # p_= 101325  * np.exp(-9.8 * orog * 0.02896968 / (288.16 * 8.314462618 ))
    # p_ = p_ - 101325
    # p_.plot()
    # plt.show()
    # mslpres = sfcpres - p_
    # mslpres.isel(time=0).plot(robust=True)
    # mslpres.isel(time=0).plot.contour(levels=30)
    # plt.show()
    pr = xr.open_dataarray(pr_filepath, chunks={'time': 140})
    pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
    pr = pr.sortby('longitude')
    pr = pr.sortby('latitude')
    pr = pr.sel(latitude=slice(-75, 60), longitude=slice(-150, 45))
    pr = pr.sel(time=timesel)
    pr = pr * 3600
    pr = pr.load()
    u = u / tcwv
    v = v / tcwv
    u.name = 'u'
    v.name = 'v'
    dt = 12
    for dt in range(1, 30):
        time1 = time.time()
        timeseq = np.arange(0, 8) + dt
        ds = xr.merge([u, v])
        ds = ds.isel(time=timeseq)
        coarse_factor = 1
        ds_coarse = ds.coarsen(latitude=coarse_factor, longitude=coarse_factor, boundary='trim').mean()
        tcwv_coarse = tcwv.isel(time=timeseq).coarsen(latitude=coarse_factor, longitude=coarse_factor,
                                                      boundary='trim').mean()
        mslpres_coarse = mslpres.isel(time=timeseq).coarsen(latitude=coarse_factor, longitude=coarse_factor,
                                                      boundary='trim').mean()
        # ds_coarse = ds
        lcs = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None)
        ftle = lcs(ds_coarse, s=1e5, resample='3H')

        lcs_local = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None)
        ftle_local = lcs(ds_coarse.isel(time=slice(-1, None)), s=1e5, resample=None)

        ftle = np.log(ftle) / 2
        ftle_local = np.log(ftle_local) * 4
        ftle_local = ftle_local.isel(time=0)
        ftle = ftle.isel(time=0)
        from skimage.filters import threshold_otsu, threshold_local
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
        block_size = 301
        local_thresh = ftle_local.copy(data=threshold_local(ftle_local.values, block_size,
                                                            offset=-.8))

        binary_local = ftle_local > local_thresh
        ftle_local_high=binary_local

        distance = binary_local
        ridges, eigmin, eigvectors = find_ridges_spherical_hessian(ftle,
                                                                   sigma=1.2,
                                                                   scheme='second_order',
                                                                   angle=20 * coarse_factor, return_eigvectors=True)

        ridges = ridges.copy(data=skeletonize(ridges.values))
        ridges.plot()
        plt.show()
        ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
                               thresholds=[1.2, 30 / coarse_factor])

        ridges = ridges.where(~xr.ufuncs.isnan(ridges), 0)

        mslpres_coarse = mslpres_coarse.interp(latitude=ridges.latitude.values,
                                            longitude=ridges.longitude.values,
                                            method='linear')
        dpdx = mslpres_coarse.isel(time=-1).differentiate('longitude')
        dpdy = mslpres_coarse.isel(time=-1).differentiate('latitude')
        u_vec = eigvectors.isel(eigvectors=1)
        v_vec = eigvectors.isel(eigvectors=0)
        v_vec.plot()
        plt.show()

        pres_gradient_parallel = np.sqrt((dpdx * v_vec)**2 + (dpdy * u_vec)**2)
        ridges_pres_grad = ridges * pres_gradient_parallel

        ridges_pres_grad = filter_ridges(ridges, ridges_pres_grad, criteria=['mean_intensity'],
                               thresholds=[50])
        ridges_bool = ridges == 1
        dist = ftle_local_high.copy(data=ndi.distance_transform_edt(~ridges_bool))

        ridges_dil = ridges.copy(data=binary_dilation(ridges.values))
        ridges_dil = ridges_dil.where(ridges_dil > 0)
        ridges = ridges.where(ridges > 0)
        ridges_pres_grad = ridges_pres_grad.where(ridges_pres_grad > 0)
        ridges_pres_grad = ridges_pres_grad.where(xr.ufuncs.isnan(ridges_pres_grad), 1)


        ridges_ = filter_ridges(ftle_local_high, ridges_dil.where(~xr.ufuncs.isnan(ridges_dil), 0),
                                criteria=['max_intensity'],
                                thresholds=[.5])
        ridges_ = ridges_ * dist.where(dist < 12)
        ridges_ = ridges_.where(xr.ufuncs.isnan(ridges_), 1)
        local_strain = ftle_local_high - ridges_.where(~xr.ufuncs.isnan(ridges_), 0)


        pr_ = pr.sel(time=ridges.time).interp(method='nearest',
                                              latitude=ridges.latitude,
                                              longitude=ridges.longitude)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        mslpres_coarse.isel(time=-1).plot.contourf(cmap='viridis', ax=ax,
                                                   levels=12, linewidth=.5, vmin=99600, vmax=102600,
                                                   add_colorbar=False)

        pr_.plot.contourf(add_colorbar=False, levels=[1, 5, 10, 15], alpha=.5, cmap=cmr.freeze_r, ax=ax)
        matplotlib.rcParams['hatch.color'] = 'blue'
        matplotlib.rcParams['hatch.linewidth'] = .3

        local_strain.plot.contourf(cmap='Reds', alpha=0,
                                  levels=[0, .5],
                                  add_colorbar=False,
                                  hatches=['',
                                           '////////'],
                                  ax=ax)
        matplotlib.rcParams['hatch.color'] = 'red'
        ridges_.where(~xr.ufuncs.isnan(ridges_), 0).plot.contourf(cmap='Reds', alpha=0,
                                                                  levels=[0, .5],
                                                                  add_colorbar=False,
                                                                  hatches=['',
                                                                           'xxxxxxx'],
                                                                  ax=ax)
        ridges.plot(add_colorbar=False, cmap='Purples', ax=ax)
        ridges_pres_grad.plot(add_colorbar=False, cmap=cmr.bubblegum, ax=ax)
        ax.coastlines(color='black')
        total_rain = np.sum(pr_)
        czs_rain = np.sum(ridges_ * pr_)
        local_strain_rain = np.sum(local_strain * pr_)
        rest = total_rain - local_strain_rain - czs_rain
        ax.text(-60, 0, str(np.round(czs_rain.values / 1000)) + ' m rain on CZs \n ' +
                str(np.round(local_strain_rain.values / 1000)) + ' m rain on LStr \n' +
                str(np.round(rest.values / 1000)) + 'm remaining \n',
                bbox={'color': 'black', 'alpha': .2},
                color='black')
        plt.savefig(f'temp_figs/fig{dt:02d}_area.png', dpi=600,
                    transparent=False, pad_inches=.2, bbox_inches='tight'
                    )
        plt.close()



        time2 = time.time()
        ellapsed = time2 - time1
        print('Ellapsed time: ' + str(np.round(ellapsed / 60)) + 'minutes')


        #
        # ftle_local = ftle_local.isel(time=0)
        # ftle_local.plot(vmin=3, vmax=7)
        # ftle.plot(vmin=3, vmax=5)
        # pr_.plot.contour(add_colorbar=False,levels=[1,5,10,15], cmap='Reds')
        # ridges.plot(add_colorbar=False)
        # plt.show()
        # xx = ftle.stack(points=['latitude', 'longitude'])
        # yy = pr_.stack(points=['latitude', 'longitude'])
        # cc = c_int.stack(points=['latitude', 'longitude'])
        # plt.scatter(xx.values, yy.values, c= cc.values)
        # plt.show()
        #
        # ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
        #                        thresholds=[1.2, 30 / coarse_factor])
        #
        # xs, ys, cs = parcel_propagation(ds_coarse.u, ds_coarse.v, timestep=-3600 * 3,
        #                                 C=tcwv_coarse,
        #                                 propdim='time', SETTLS_order=4, return_traj=True)
        # ftle.plot(vmin=2, vmax=5)
        # ridges.plot(add_colorbar=False)
        # plt.show()
        # c_int = cs.mean('time')
        # boundaries = find_area(ftle, eigvectors, ridges, qdpt=c_int, qsat=60)  # qsat kind of arbitrary for now
        # boundaries = boundaries.copy(data=gaussian_filter(boundaries, sigma=2))
        # boundaries = boundaries.where(boundaries>.5)
        # pr_ = pr.sel(time=ridges.time).interp(method='nearest',
        #                              latitude=ridges.latitude,
        #                              longitude=ridges.longitude)
        # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        # pr_.plot(ax=ax, add_colorbar=False, cmap='Blues', vmax=10)
        # ridges.plot(add_colorbar=False, ax=ax)
        # boundaries.plot(alpha=.5, ax=ax, cmap='Reds', add_colorbar=False)
        # pr_czs = pr_ * boundaries
        # total_area = pr_.copy(data=np.ones(shape=pr_.shape))
        # frac_area = ( total_area * boundaries ).sum()/total_area.sum()
        # ratio_prec = pr_czs.sum() / pr_.sum()
        # ax.text(-40, 0, str(np.round(ratio_prec.values* 100)) + ' % \n rain on CZs \n ' +
        #         str(np.round(frac_area.values*100)) + ' % area of CZs')
        # ax.coastlines()
        # plt.savefig(f'temp_figs/fig{dt:02d}_wet.png', dpi=600,
        #         transparent=True, pad_inches=.2, bbox_inches='tight'
        #         )
        # plt.close()

    #
    # plt.show()
    # ftle.plot()
    # u_vec = eigvectors.isel(eigvectors=1).values * eigmin.values
    # v_vec = eigvectors.isel(eigvectors=0).values * eigmin.values
    #
    # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    # ftle.plot(add_colorbar=True, ax=ax, transform=ccrs.PlateCarree())
    # ridges.plot(add_colorbar=False, ax=ax, transform=ccrs.PlateCarree())
    # ax.quiver(eigvectors.longitude.values,
    #            eigvectors.latitude.values,
    #            u_vec, v_vec,scale=1e-8, transform=ccrs.PlateCarree())
    # ax.coastlines()
    # plt.savefig('eigvectors.png', dpi=600,
    #             transparent=True, pad_inches=.2, bbox_inches='tight'
    #             )
    # plt.close()
