import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LagrangianCoherence.LCS import LCS
import cmasher as cmr
import cartopy.crs as ccrs
from LagrangianCoherence.LCS import trajectory


def ideal_saddle(lat_min, lat_max, lon_min, lon_max, dx, dy, nt, max_intensity=10):
    """
    Method to initialize an ideal vortex
    Parameters
    ----------
    nx grid width
    ny grid height

    Returns: vorticity numpy array
    -------
    """
    lats = np.arange(lat_min, lat_max, dy)
    lons = np.arange(lon_min, lon_max, dx)
    nx = lons.shape[0]
    ny = lats.shape[0]
    u = np.zeros([ny, nx, nt])
    v = np.zeros([ny, nx, nt])
    coords = {
        'lon': lons,
        'lat': lats,
        'time': pd.date_range('2000-01-01', periods=nt, freq='6H')
    }

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                u[y, x, t] = max_intensity * y / ny - .5 * max_intensity
                v[y, x, t] = max_intensity * x / nx - .5 * max_intensity
    u = xr.DataArray(u, dims=['lat', 'lon', 'time'], coords=coords)
    v = xr.DataArray(v, dims=['lat', 'lon', 'time'], coords=coords)

    return u, v


def ideal_vortex(lat_min, lat_max, lon_min, lon_max, dx, dy, nt, max_intensity=10, radius=5, center=None, u_c=0, v_c=0):
    """
    Method to initialize an ideal vortex
    Parameters
    ----------
    nx grid width
    ny grid height

    Returns: vorticity numpy array
    -------
    """
    lats = np.arange(lat_min, lat_max, dy)
    lons = np.arange(lon_min, lon_max, dx)
    nx = lons.shape[0]
    ny = lats.shape[0]
    u = np.zeros([ny, nx, nt])
    theta = np.zeros([ny, nx, nt])
    v = np.zeros([ny, nx, nt])
    coords = {
        'lon': lons,
        'lat': lats,
        'time': pd.date_range('2000-01-01', periods=nt, freq='6H')
    }

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                new_x = lons[x] - center[0] - u_c * t
                new_y = lats[y] - center[1] - v_c * t
                distance = np.sqrt(new_x ** 2 + new_y ** 2)

                theta[y, x, t] = np.arccos(new_y / (distance + 1e-8))
                if distance > radius:
                    mag = max_intensity * radius ** 2 / (2 * distance)
                else:
                    mag = max_intensity * 0.5 * distance

                u[y, x, t] = np.cos(theta[y, x, t]) * mag
                if new_x < 0:
                    v[y, x, t] = np.sin(theta[y, x, t]) * mag
                else:
                    v[y, x, t] = np.sin(theta[y, x, t] + np.pi) * mag

    u = xr.DataArray(u, dims=['lat', 'lon', 'time'], coords=coords)
    v = xr.DataArray(v, dims=['lat', 'lon', 'time'], coords=coords)
    theta = xr.DataArray(theta, dims=['lat', 'lon', 'time'], coords=coords)

    return u, v, theta


vortex_config_equator = dict(
    lat_min=-40,
    lat_max=40,
    lon_min=-80,
    lon_max=0,
    dx=0.5,
    dy=0.5,
    u_c=0.5,
    v_c=0.5,
    nt=20,
    radius=4,
    max_intensity=6,
    center=[-40, 0]
)
vortex_config_subtropical = dict(
    lat_min=-10,
    lat_max=70,
    lon_min=-80,
    lon_max=0,
    dx=1,
    dy=1,
    nt=20,
    u_c=0,
    v_c=0,
    radius=2,
    max_intensity=6,
    center=[-40, 30]
)
subdomain = {
    'latitude': slice(-20, 20),
    'longitude': slice(-60, -20),
}

saddle_config = dict(
    lat_min=-70,
    lat_max=-10,
    lon_min=-70,
    lon_max=-10,
    dx=1,
    dy=1,
    nt=10,
    max_intensity=10,
)

vortex_config = vortex_config_equator
u, v, theta = ideal_vortex(**vortex_config)
# u, v = ideal_saddle(**saddle_config)
u.name = 'u'
v.name = 'v'

ds = xr.merge([u, v])
u_plot = u.isel(time=-1).values
v_plot = v.isel(time=-1).values
x_plot = v.isel(time=-1).lon.values
y_plot = v.isel(time=-1).lat.values

ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
x, y = trajectory.parcel_propagation(ds.u, ds.v,
                                     timestep=-6 * 3600,
                                     propdim='time',
                                     SETTLS_order=8,
                                     copy=True,
                                     s=None)

lcs = LCS.LCS(lcs_type='repelling', timestep=-6 * 3600, timedim='time', SETTLS_order=8, )
ftle = lcs(ds)
ftle = np.log(np.sqrt(ftle)) / vortex_config['nt']

np.abs(u).sel(lon=u.lon.mean(), method='nearest').isel(time=0).plot()
plt.show()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
x.plot.contourf(levels=20, cmap=cmr.iceburn, vmin=x.values.min(), vmax=x.values.max(), ax=ax, add_colorbar=True)
ax.streamplot(x=x_plot, y=y_plot, u=u_plot, v=v_plot)
ax.coastlines()
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
ax.add_artist(draw_circle, )
plt.show()
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
y.plot.contourf(levels=20, cmap=cmr.iceburn, vmin=y.values.min(), vmax=y.values.max(), ax=ax, add_colorbar=True)
ax.streamplot(x=x_plot, y=y_plot, u=u_plot, v=v_plot)
ax.coastlines()
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
ax.add_artist(draw_circle, )
plt.show()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
ftle.isel(time=0).plot(cmap=cmr.rainforest, vmin=0, ax=ax, vmax=0.15)
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
ax.coastlines()
ax.add_artist(draw_circle)
plt.show()

from skimage.filters import frangi, hessian

ridges = ftle.groupby('time').apply(hessian)
ftle.plot(cmap=cmr.rainforest, vmin=0)
ridges.isel(time=0).plot.contour()
plt.show()

div = u.differentiate('lon').isel(time=0) + v.differentiate('lat').isel(time=0)
div.plot()
plt.show()
from scipy.interpolate import RectSphereBivariateSpline

lats = np.radians((ds.latitude + 90) % 90).values
lons = np.radians((ds.longitude + 180) % 180).values
data = ds.u.isel(time=0).values
positions_y, positions_x = np.meshgrid(lats, lons)

interpolator_y = RectSphereBivariateSpline(lats, lons, data)
va = interpolator_y(lats + 1e-4, lons)
plt.imshow(va)
plt.show()
plt.imshow(data)
plt.show()

va.shape()
va = interpolator_y(lats + 1e-4, lons + 1e-4)
plt.imshow(va)
plt.show()
plt.imshow(data)
plt.show()

theta = np.linspace(0., np.pi, 7)
phi = np.linspace(0., 2 * np.pi, 9)
data = np.empty((theta.shape[0], phi.shape[0]))
data[:, 0], data[0, :], data[-1, :] = 0., 0., 0.
data[1:-1, 1], data[1:-1, -1] = 1., 1.
data[1, 1:-1], data[-2, 1:-1] = 1., 1.
data[2:-2, 2], data[2:-2, -2] = 2., 2.
data[2, 2:-2], data[-3, 2:-2] = 2., 2.
data[3, 3:-2] = 3.
data = np.roll(data, 4, 1)

lats, lons = np.meshgrid(theta, phi)
from scipy.interpolate import SmoothSphereBivariateSpline

lut = RectSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                data.T.ravel(), s=3.5)
fine_lats = np.linspace(0., np.pi, 70)
fine_lons = np.linspace(0., 2 * np.pi, 90)
data_orig = lut(theta, phi)
data_smth = lut(fine_lats, fine_lons)
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(data, interpolation='nearest')
ax2 = fig.add_subplot(132)
ax2.imshow(data_orig, interpolation='nearest')
ax3 = fig.add_subplot(133)
ax3.imshow(data_smth, interpolation='nearest')
plt.show()
