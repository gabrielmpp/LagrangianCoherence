import numpy as np
from skimage.feature import hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def find_ridges_spherical_hessian(da, sigma=.5, scheme='first_order',
                                  tolerance_threshold=0.0005e-3, return_eigvectors=False):
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

    # Initializing
    earth_r = 6371000
    x = da.longitude.copy() * np.pi / 180
    y = da.latitude.copy() * np.pi / 180
    dx = x.diff('longitude') * earth_r * np.cos(y)
    dy = y.diff('latitude') * earth_r
    dx_scaling = 2 * da.longitude.diff('longitude').values[0]  # grid spacing for xr.differentiate
    dy_scaling = 2 * da.latitude.diff('latitude').values[0]  # grid spacing
    # Calc derivatives
    if scheme == 'second_order':
        ddadx = dx_scaling * da.differentiate('longitude') / dx
        ddady = dy_scaling * da.differentiate('latitude') / dy
        d2dadx2 = dx_scaling * ddadx.differentiate('longitude') / dx
        d2dadxdy = dy_scaling * ddadx.differentiate('latitude') / dy
        d2dady2 = dx_scaling * ddady.differentiate('latitude') / dy
        d2dadydx = d2dadxdy.copy()
    elif scheme == 'first_order':
        ddadx = da.diff('longitude') / dx
        ddady = da.diff('latitude') / dy
        d2dadx2 = ddadx.diff('longitude') / dx
        d2dadxdy = ddadx.diff('latitude') / dy
        d2dady2 = ddady.diff('latitude') / dy
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

        dt_angle = np.dot(eigvector, grad_vals[:, i]) #/ (np.linalg.norm(eigvector) *
                                                        #                    np.linalg.norm(grad_vals[:, i]))
        # dt_angle = .5*np.pi - np.arccos(np.abs(dt_angle))
        val_list.append(dt_angle)
        eigmin_list.append(eig[0][np.argmax(np.abs(eig[0]))])
        eigvector_list.append(eigvector)
        eigvector_min_list.append(eigvector_min)


    eigvectors=hessian.isel(elements=[1, 2]).copy(data=np.array(eigvector_list).T).rename(elements='eigvectors').unstack()
    angle = 180/np.pi*np.arctan(eigvectors.isel(eigvectors=0) / eigvectors.isel(eigvectors=1)).T

    eigvectors_min=hessian.isel(elements=[1, 2]).copy(data=np.array(eigvector_min_list).T).rename(elements='eigvectors').unstack()
    dt_prod = hessian.isel(elements=0).drop('elements').copy(data=val_list).unstack()
    dt_prod_ = dt_prod.copy()
    eigmin = hessian.isel(elements=0).drop('elements').copy(data=eigmin_list).unstack()
    eigvectors = eigvectors.where(eigmin < 0, 0)

    dt_prod = dt_prod.where(np.abs(dt_prod_) <= tolerance_threshold, 0)
    dt_prod = dt_prod.where(np.abs(dt_prod_) > tolerance_threshold, 1)
    dt_prod = dt_prod.where(np.sign(eigmin) == -1, 0)
    angle = angle*dt_prod.where(dt_prod > 0)
    # shear = shear.where(dt_prod == 1)
    # rd = np.sqrt(np.abs(shear) * 86400 / da)
    # rd.plot(robust=True)
    # plt.show()
    # rd.plot.hist()
    # plt.show()
    # plt.log(True)
    if return_eigvectors:
        return dt_prod.unstack().transpose(..., *original_dim_order), eigmin.unstack().transpose(..., *original_dim_order), \
               eigvectors.unstack().transpose(..., *original_dim_order), gradient.unstack().transpose(..., *original_dim_order), \
               angle.transpose(..., *original_dim_order)
    else:
        return dt_prod.unstack().transpose(..., *original_dim_order), eigmin.unstack().transpose(..., *original_dim_order),


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
