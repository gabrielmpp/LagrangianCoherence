import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
import pandas as pd
from copy import deepcopy
from numba import jit
from scipy.ndimage import map_coordinates
import windspharm


def xr_map_coordinates(da, new_x, new_y, isglobal=True, order=1):
    da = da.copy().transpose('latitude', 'longitude')
    new_x = new_x.copy()
    new_y = new_y.copy()
    if isinstance(new_y, (xr.DataArray, xr.Dataset)):
        new_y = new_y.values
    if isinstance(new_x, (xr.DataArray, xr.Dataset)):
        new_x = new_x.values
    x_size = da.longitude.values.shape[0]
    y_size = da.latitude.values.shape[0]
    new_x = x_size * (new_x - da.longitude.min().values) / (da.longitude.max().values - da.longitude.min().values)
    new_y = y_size * (new_y - da.latitude.min().values) / (da.latitude.max().values - da.latitude.min().values)
    if isglobal:
        da_except_poles = da.isel(latitude=slice(order, -order))
        idxs = np.arange(order, da.latitude.values.shape[0]-order)
        da_interp = da_except_poles.copy(data=map_coordinates(da.values,
                                                              np.array([new_y[idxs, :].ravel(),
                                                                        new_x[idxs, :].ravel()]),
                                                              order=order,
                                                              mode='wrap').reshape(da_except_poles.shape))
        idxs_1 = np.arange(0, order)
        idxs_2 = np.arange(-order, 0)
        pole_idxs = np.hstack([idxs_1, idxs_2])
        da_poles = da.isel(latitude=pole_idxs)
        da_interp_poles = da_poles.copy(data=map_coordinates(da.values,
                                                             np.array([new_y[pole_idxs, :].ravel(),
                                                                       new_x[pole_idxs, :].ravel()]),
                                                             order=1,
                                                             mode='constant').reshape(da_poles.shape))

        da_interp = xr.concat([da_interp_poles, da_interp], dim='latitude').sortby('latitude')
    else:
        da_interp = da.copy(data=map_coordinates(da.values,
                                                 np.array([new_y.ravel(),
                                                           new_x.ravel()]),
                                                 order=1,
                                                 mode='wrap').reshape(da_except_poles.shape))
    return da_interp



def find_ridges_spherical_hessian(da, sigma=.5, scheme='first_order',
                                  tolerance_threshold=0.0005e-3, return_eigvectors=False,
                                  isglobal=True):
    """
    Method to in apply a Hessian filter in spherical coordinates
    Parameters
    ----------
    sigma - float, smoothing intensity
    da - xarray.dataarray
    scheme - str, 'first_order' for x[i+1] - x[i] and second order for x[i+1] - x[i-1]
    tolerance_threshold - tolerance for the FTLE gradient across the ridge
    Returns
    -------
    Filtered array

    """
    da = da.copy()
    original_dim_order = deepcopy(da.dims)
    da = da.transpose('longitude', 'latitude', ...)
    da = da.sortby('latitude')
    da = da.sortby('longitude')

    # Gaussian filter
    if isinstance(sigma, (float, int)):
        da = da.copy(data=gaussian_filter(da, sigma=sigma))

    ddadx = derivative_spherical_coords(da, dim=1, isglobal=isglobal)
    ddady = derivative_spherical_coords(da, dim=0, isglobal=isglobal)
    d2dadx2 = derivative_spherical_coords(ddadx, dim=1, isglobal=isglobal)
    d2dady2 = derivative_spherical_coords(ddady, dim=0, isglobal=isglobal)
    d2dadxdy = derivative_spherical_coords(ddadx, dim=0, isglobal=isglobal)
    d2dadydx = d2dadxdy.copy()
    # Assembling Hessian array
    gradient = xr.concat([ddadx, ddady],
                         dim=pd.Index(['ddadx', 'ddady'],
                                      name='elements'))
    hessian = xr.concat([d2dadx2, d2dadxdy, d2dadydx, d2dady2],
                        dim=pd.Index(['d2dadx2', 'd2dadxdy', 'd2dadydx', 'd2dady2'],
                                     name='elements'))
    hessian = hessian.stack({'points': ['longitude', 'latitude']})
    gradient = gradient.stack({'points': ['longitude', 'latitude']})
    hessian = hessian.where(np.abs(hessian) != np.inf, 0)
    hessian = hessian.where(~xr.ufuncs.isnan(hessian), 0)
    hessian = hessian.dropna('points', how='any')
    gradient = gradient.sel(points=hessian.points)
    grad_vals = gradient.transpose(..., 'points').values
    hess_vals = hessian.transpose(..., 'points').values

    hess_vals = hess_vals.reshape([2, 2, hessian.shape[-1]])
    val_list = []
    eigmin_list = []
    eigvector_list = []
    eigvector_min_list = []
    for i in range(hess_vals.shape[-1]):
        # print(str(100 * i / hess_vals.shape[-1]) + ' %')
        eig = np.linalg.eig(hess_vals[:, :, i])
        eigvector = eig[1][np.argmin(eig[0])]
        # eigvector = eig[1][np.argmin(eig[0])]

        eigvector_min = eig[1][np.argmin(np.abs(eig[0]))]

        # eigenvetor of smallest eigenvalue
        # eigvector = eigvector/np.max(np.abs(eigvector))  # normalizing the eigenvector to recover t hat

        dt_angle = np.dot(eigvector, grad_vals[:, i])  / (np.linalg.norm(eigvector) * np.linalg.norm(grad_vals[:, i]))
        # dt_angle = .5*np.pi - np.arccos(np.abs(dt_angle))
        val_list.append(dt_angle)
        eigmin_list.append(eig[0][np.argmax(np.abs(eig[0]))])
        eigvector_list.append(eigvector)
        eigvector_min_list.append(eigvector_min)

    eigvectors = hessian.isel(elements=[1, 2]).copy(data=np.array(eigvector_list).T).rename(
        elements='eigvectors').unstack()
    angle = 180 / np.pi * np.arctan(eigvectors.isel(eigvectors=0) / eigvectors.isel(eigvectors=1)).T

    eigvectors_min = hessian.isel(elements=[1, 2]).copy(data=np.array(eigvector_min_list).T).rename(
        elements='eigvectors').unstack()
    dt_prod = hessian.isel(elements=0).drop('elements').copy(data=val_list).unstack()
    dt_prod_ = dt_prod.copy()
    eigmin = hessian.isel(elements=0).drop('elements').copy(data=eigmin_list).unstack()
    eigvectors = eigvectors.where(eigmin < 0, 0)

    dt_prod = dt_prod.where(np.abs(dt_prod_) <= tolerance_threshold, 0)
    dt_prod = dt_prod.where(np.abs(dt_prod_) > tolerance_threshold, 1)
    dt_prod = dt_prod.where(np.sign(eigmin) == -1, 0)

    # angle = angle * dt_prod.where(dt_prod > 0)
    # shear = shear.where(dt_prod == 1)
    # rd = np.sqrt(np.abs(shear) * 86400 / da)
    # rd.plot(robust=True)
    # plt.show()
    # rd.plot.hist()
    # plt.show()
    # plt.log(True)
    if return_eigvectors:
        return dt_prod.unstack().transpose(..., *original_dim_order), eigmin.unstack().transpose(...,
                                                                                                 *original_dim_order), \
               dt_prod_.unstack().transpose(..., *original_dim_order), \
               eigvectors.unstack().transpose(..., *original_dim_order), gradient.unstack().transpose(...,
                                                                                                      *original_dim_order), \
               angle.transpose(..., *original_dim_order)
    else:
        return dt_prod.unstack().transpose(..., *original_dim_order), eigmin.unstack().transpose(...,
                                                                                                 *original_dim_order)


def latlonsel(array, lat, lon, latname='lat', lonname='lon'):
    """
    Function to crop array based on lat and lon intervals given by slice or list.
    This function is able to crop across cyclic boundaries.

    :param array: xarray.Datarray
    :param lat: list or slice (min, max)
    :param lon: list or slice(min, max)
    :return: cropped array
    """
    assert latname in array.coords, f"Coord. {latname} not present in array"
    assert lonname in array.coords, f"Coord. {lonname} not present in array"

    if isinstance(lat, slice):
        lat1 = lat.start
        lat2 = lat.stop
    elif isinstance(lat, list):
        lat1 = lat[0]
        lat2 = lat[-1]
    if isinstance(lon, slice):
        lon1 = lon.start
        lon2 = lon.stop
    elif isinstance(lon, list):
        lon1 = lon[0]
        lon2 = lon[-1]

    lonmask = (array[lonname] < lon2) & (array[lonname] > lon1)
    latmask = (array[latname] < lat2) & (array[latname] > lat1)
    array = array.where(lonmask, drop=True).where(latmask, drop=True)
    return array


@jit(nopython=True)
def fourth_order_derivative(arr: np.ndarray, dim=0, isglobal=True):
    """
    2D numpy array with dims [lat, lon]
    :param arr:
    :return:
    """
    # assert isinstance(arr, np.ndarray), 'Input must be numpy array'
    output = np.zeros_like(arr)

    if dim == 0:
        ysize = np.shape(arr)[0]
        for lat_idx in range(2, np.shape(arr)[0] - 2):
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (4 / 3) * (arr[(lat_idx + 1), lon_idx] -
                                                      arr[(lat_idx - 1), lon_idx]) / 2 \
                                           - (1 / 3) * (arr[(lat_idx + 2), lon_idx] -
                                                        arr[(lat_idx - 2), lon_idx]) / 4

        #  First order uncentered derivative for points close to the poles
        for lat_idx in [0, 1]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[(lat_idx + 1), lon_idx] -
                                            arr[lat_idx, lon_idx]) / 2
        for lat_idx in [-1, -2]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[lat_idx, lon_idx] -
                                            arr[lat_idx - 1, lon_idx]) / 2
    elif dim == 1:
        xsize = np.shape(arr)[1]
        if isglobal:
            for lat_idx in range(np.shape(arr)[0]):

                for lon_idx in range(np.shape(arr)[1]):

                    output[lat_idx, lon_idx] = (4 / 3) * (arr[lat_idx, (lon_idx + 1) % xsize] -
                                                          arr[lat_idx, (lon_idx - 1) % xsize]) / 2 \
                                               - (1 / 3) * (arr[lat_idx, (lon_idx + 2) % xsize] -
                                                            arr[lat_idx, (lon_idx - 2) % xsize]) / 4
        else:
            for lat_idx in range(np.shape(arr)[0]):
                for lon_idx in range(2, np.shape(arr)[1] - 2):
                    output[lat_idx, lon_idx] = (4 / 3) * (arr[lat_idx, (lon_idx + 1)] -
                                                          arr[lat_idx, (lon_idx - 1)]) / 2 \
                                               - (1 / 3) * (arr[lat_idx, (lon_idx + 2)] -
                                                            arr[lat_idx, (lon_idx - 2)]) / 4
            #  First order uncentered derivative for points close to the bondaries
            for lon_idx in [0, 1]:
                for lat_idx in range(np.shape(arr)[0]):
                    output[lat_idx, lon_idx] = (arr[lat_idx, lon_idx+1] -
                                                arr[lat_idx, lon_idx]) / 2
            for lon_idx in [-1, -2]:
                for lat_idx in range(np.shape(arr)[0]):
                    output[lat_idx, lon_idx] = (arr[lat_idx, lon_idx] -
                                                arr[lat_idx, lon_idx-1]) / 2
    return output


def derivative_spherical_coords(da, dim=0, isglobal=True):
    EARTH_RADIUS = 6371000  # m
    da = da.sortby('latitude')
    da = da.sortby('longitude')
    da = da.transpose('latitude', 'longitude')
    x = da.longitude.copy() * np.pi / 180
    y = da.latitude.copy() * np.pi / 180
    dx = (np.pi/180) * (da.longitude.values[1] - da.longitude.values[0]) * EARTH_RADIUS * np.cos(y)
    dy = (np.pi/180) * (da.latitude.values[1] - da.latitude.values[0]) * EARTH_RADIUS

    deriv = fourth_order_derivative(da.values.astype('float32'), dim=dim, isglobal=isglobal)
    deriv = da.copy(data=deriv)

    if dim == 0:
        deriv = deriv / dy
    elif dim == 1:
        deriv = deriv / dx
    else:
        raise ValueError('Dim must be either 0 or 1.')
    return da.copy(data=deriv)



@jit(nopython=True)
def harvesine(lon1, lat1, lon2, lat2):
    rad = np.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (np.sin(dlat / 2)) ** 2 + np.cos(lat1 * rad) * \
        np.cos(lat2 * rad) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


@jit(nopython=True)
def Inverse_weighted_interpolation(x, y, z, xi, yi,power=2):
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, power)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        # xyzi = [xi[p], yi[p], u]
        xyzi = u
        lstxyzi.append(xyzi)
    return lstxyzi


def xr_idx_interp(ds_y_for_spline, lon, lat, p=2):
    """
    https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
    :param ds_y_for_spline:
    :param lon:
    :param lat:
    :return:
    """
    ds_y_for_spline = ds_y_for_spline.load()
    print('interp idx')
    Lon, Lat = np.meshgrid(lon, lat)

    out_shape = Lon.shape

    Lon = Lon.ravel()
    Lat = Lat.ravel()

    def interp_idw(dd,Lon,Lat, out_shape,lat,lon,p):

        ds_interp = Inverse_weighted_interpolation(dd.longitude.values,
                                                   dd.latitude.values,
                                                   dd.values,
                                                   Lon,
                                                   Lat,p)

        ds_interp = xr.DataArray(np.array(ds_interp).reshape(out_shape), dims=['latitude', 'longitude'],
                                 coords={'longitude':lon,
                                         'latitude':lat})
        return ds_interp

    ds_interp_grid = interp_idw(ds_y_for_spline, Lon, Lat, out_shape, lat, lon, p)
    return ds_interp_grid


