import numpy as np
from skimage.feature import hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter
import xarray as xr
import pandas as pd


def find_ridges_spherical_hessian(da, sigma=.5, scheme='first_order'):
    """
    Method to in apply a Hessian filter in spherical coordinates
    Parameters
    ----------
    sigma - float, smoothing intensity
    da - xarray.dataarray
    scheme - str, 'first_order' for x[i+1] - x[i] and second order for x[i+1] - x[i-1]

    Returns
    -------
    Filtered array

    """
    da = da.copy()
    # Gaussian filter
    if isinstance(sigma, (float, int)):
        da = da.copy(data=gaussian_filter(da, sigma=sigma))

    # Initializing
    earth_r = 6371000
    x = da.longitude.copy() * np.pi/180
    y = da.latitude.copy() * np.pi/180
    dx = x.diff('longitude') * earth_r * np.cos(y)
    dy = y.diff('latitude') * earth_r
    dx_scaling = 2 * da.longitude.diff('longitude').values[0]  # grid spacing for xr.differentiate
    dy_scaling = 2 * da.latitude.diff('latitude').values[0]  # grid spacing
    print(1)
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
    print(2)
    hessian = xr.concat([d2dadx2, d2dadxdy, d2dadydx, d2dady2],
                            dim=pd.Index(['d2dadx2', 'd2dadxdy', 'd2dadydx', 'd2dady2'],
                                         name='elements'))
    hessian = hessian.stack({'points': ['latitude', 'longitude']})
    # hessian = hessian.dropna('points', how='any')

    # Finding norm
    print(3)
    norm = hessian_matrix_eigvals([hessian.sel(elements='d2dadx2').values,
                                   hessian.sel(elements='d2dadxdy').values,
                                   hessian.sel(elements='d2dady2').values])
    norm_max = hessian.isel(elements=0).drop('elements').copy(data=norm[1, :]).unstack()

    return norm_max


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
