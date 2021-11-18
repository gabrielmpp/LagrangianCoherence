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


def rotating_saddle(lat_min, lat_max, lon_min, lon_max, dx, dy, nt, max_intensity=10, radius=5, center=None, u_c=0, v_c=0):
    """
    Method to initialize a rotating saddle
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
        'longitude': lons,
        'latitude': lats,
        'time': pd.date_range('2000-01-01', periods=nt, freq='6H')
    }

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                new_x = (lons[x] - center[0])/180
                new_y = (lats[y] - center[1])/90

                u[y, x, t] = np.sqrt(2)*max_intensity*(np.sin(4*t/nt) * new_x + (2 + np.cos(4*t/nt)) * new_y)
                v[y, x, t] = np.sqrt(2)*max_intensity*((-2 * np.cos(4*t/nt)) * new_x - np.sin(4*t/nt) * new_y)
    old_u = u.copy()
    old_v = v.copy()
    # u[np.where((old_u**2 + old_v**2)**0.5 > max_intensity/2)] = 0
    # v[np.where((old_u ** 2 + old_v ** 2) ** 0.5 > max_intensity/2)] = 0
    u = xr.DataArray(u, dims=['latitude', 'longitude', 'time'], coords=coords)
    v = xr.DataArray(v, dims=['latitude', 'longitude', 'time'], coords=coords)

    return u, v


def shear_flow(lat_min, lat_max, lon_min, lon_max, dx, dy, nt, max_intensity=10, center=None, v_c=0):
    """
    Method to initialize a rotating saddle
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
        'longitude': lons,
        'latitude': lats,
        'time': pd.date_range('2000-01-01', periods=nt, freq='6H')
    }

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                new_x = (lons[x] - center[0])/180
                new_y = (lats[y] - center[1])/90

                # u[y, x, t] = np.cos(lats[y]*np.pi/180)
                u[y, x, t] = max_intensity
                v[y, x, t] = 0
    old_u = u.copy()
    old_v = v.copy()
    # u[np.where((old_u**2 + old_v**2)**0.5 > max_intensity/2)] = 0
    # v[np.where((old_u ** 2 + old_v ** 2) ** 0.5 > max_intensity/2)] = 0
    u = xr.DataArray(u, dims=['latitude', 'longitude', 'time'], coords=coords)
    v = xr.DataArray(v, dims=['latitude', 'longitude', 'time'], coords=coords)

    return u, v


def ideal_vortex(lat_min, lat_max, lon_min, lon_max, dx, dy, nt,
                 max_intensity=10, radius=5, center=None, u_c=0, v_c=0,
                 diag_factor=0, basic_zonal=2,
                 k=0):
    """
    Method to initialize an ideal vortex
    Parameters
    ----------

    k
    diag_factor
    v_c
    u_c
    center
    radius
    nt
    dy
    dx
    lon_max
    lon_min
    lat_max
    lat_min
    max_intensity
    -------

    """
    evap_center = [-60, 0]
    evap_rate = 10
    lats = np.arange(lat_min, lat_max, dy)
    lons = np.arange(lon_min, lon_max, dx)
    nx = lons.shape[0]
    ny = lats.shape[0]
    u = np.zeros([ny, nx, nt])
    evap = np.zeros([ny, nx, nt])
    wv = np.zeros([ny, nx, nt])
    theta = np.zeros([ny, nx, nt])
    v = np.zeros([ny, nx, nt])
    coords = {
        'longitude': lons,
        'latitude': lats,
        'time': pd.date_range('2000-01-01', periods=nt, freq='6H')
    }

    is_not_initial_time = 0
    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                new_x = lons[x] - center[0] - u_c * t
                if k > 0:
                    new_y = lats[y] - center[1] - v_c * np.sin(k*2*np.pi*t/nt)
                elif k ==0:
                    new_y = lats[y] - center[1] - v_c * t
                else:
                    raise ValueError('Meridional wavenumber k must be greater than zero.')
                distance = np.sqrt(new_x ** 2 + new_y ** 2)
                distance_evap = np.sqrt((lons[x] - evap_center[0]) ** 2 +
                                        (lats[y] - evap_center[1]) ** 2)
                evap[y, x, t] = evap_rate / (distance_evap + 1)**2
                wv[y, x, t] = wv[y, x, t - 1*is_not_initial_time] + evap[y, x, t]

                theta[y, x, t] = np.arccos(new_y / (distance + 1e-8))
                if distance > radius:
                    mag = max_intensity * radius ** 2 / (2 * distance)
                else:
                    mag = max_intensity * 0.5 * distance

                u[y, x, t] = np.cos(theta[y, x, t]) * mag + basic_zonal
                if new_x < 0:
                    v[y, x, t] = np.sin(theta[y, x, t] ) * mag
                else:
                    v[y, x, t] = np.sin(theta[y, x, t]  + np.pi) * mag
        is_not_initial_time = 1

    u = xr.DataArray(u, dims=['latitude', 'longitude', 'time'], coords=coords)
    v = xr.DataArray(v, dims=['latitude', 'longitude', 'time'], coords=coords)
    wv = xr.DataArray(wv, dims=['latitude', 'longitude', 'time'], coords=coords)
    evap = xr.DataArray(evap, dims=['latitude', 'longitude', 'time'], coords=coords)

    return u, v


vortex_config_equator = {'lat_min': -88, 'lat_max': 89, 'lon_min': -180,'lon_max': 180, 'dx': 2,
                             'dy': 2, 'u_c': 0, 'k': 4, 'diag_factor': 1,
                         'v_c': 0, 'nt': 30, 'radius': 2, 'max_intensity': 40, 'center': [-55, -20]}

shear_flow_config = {'lat_min': -40, 'lat_max': 40, 'lon_min': -60, 'lon_max': 20, 'dx':1, 'dy': 1,
                         'v_c': 0.2, 'nt': 30,  'max_intensity': 1, 'center': [-20, 0]}
subdomain={'latitude': slice(None, None), 'longitude': slice(None, None)}


vortex_config_subtropical = {'lat_min': -88, 'lat_max': 89, 'lon_min': -180,'lon_max': 180, 'dx': 2,
                             'dy': 2, 'u_c': 0, 'k': 0, 'diag_factor': 1,
                         'v_c': 0, 'nt': 8, 'radius': 2, 'max_intensity': 60, 'center': [-55, -20],
                             'basic_zonal': 0}
subdomain = {
    'latitude': slice(-20, 20),
    'longitude': slice(-60, -20),
}

saddle_config = {'lat_min': -70, 'lat_max': -10, 'lon_min': -70, 'lon_max': -10, 'dx': 1, 'dy': 1, 'nt': 10,
                 'max_intensity': 10}


vortex_type = 'unsteady'

if vortex_type == 'steady':
    u, v = ideal_vortex(**vortex_config_equator)
    vortex_config = vortex_config_equator
elif vortex_type == 'unsteady':
    u, v = ideal_vortex(**vortex_config_subtropical)
    vortex_config = vortex_config_subtropical
elif vortex_type == 'twin':
    # ---- Twin vortices ---- #
    from copy import deepcopy

    vortex_config_1 = deepcopy(vortex_config_equator)
    vortex_config_2 = deepcopy(vortex_config_equator)
    vortex_config_1['center'] = [-40, -20]
    vortex_config_2['center'] = [-40, 21]
    u1, v1, wv1, evap1 = ideal_vortex(**vortex_config_1)
    u2, v2, wv2, evap2 = ideal_vortex(**vortex_config_2)
    u, v = - u1 + u2, - v1 + v2

u.name = 'u'
v.name = 'v'

ds = xr.merge([u, v])

ds.isel(time=0).plot.quiver('longitude', 'latitude', 'u', 'v')
plt.show()
# ---- Computing departure points and FTLE ---- #

x_dye, y_dye = trajectory.parcel_propagation(ds.u, ds.v,
                                     timestep=-6 * 3600,
                                     propdim='time',
                                     SETTLS_order=4,
                                     copy=True,
                                     # C=wv,
                                     # Srcs=evap,
                                     return_traj=True,
                                     cyclic_xboundary=True)
x, y = trajectory.parcel_propagation(ds.u, ds.v,
                                     timestep=6 * 3600,
                                     propdim='time',
                                     SETTLS_order=2,
                                     copy=True,
                                     # C=wv,
                                     # Srcs=evap,
                                     return_traj=True,
                                     cyclic_xboundary=True)
rcs = LCS.LCS(timestep=6 * 3600, timedim='time', SETTLS_order=4 )
ftle_r = rcs(ds.copy(), isglobal=True)
ftle_r =np.log(ftle_r) / 2
ftle_r = ftle_r.where(~xr.ufuncs.isnan(ftle_r), 0)


acs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, )
ftle_a = acs(ds.copy(), isglobal=True, )
ftle_a = np.log(ftle_a) / 2

(y_dye.isel(time=0) - y_dye.isel(time=-1)).plot.contourf(levels=20)
plt.show()
(x_dye.isel(time=-1) ).plot.contourf(levels=20)
plt.show()
ftle_a.isel(time=0).plot.contourf(levels=20)
plt.show()

# ---- Plots ---- #

# Basic vortex details

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
np.sqrt(u**2 + v**2).isel(time=0).plot.contourf(levels=20, ax=ax, cmap=cmr.rainforest)
ax.set_title('Wind speed (m/s)')
ax.coastlines(color='white')
plt.savefig(f'figs/{vortex_type}_windspeed.png')
plt.close()

np.sqrt(v**2).isel(time=0).sel(latitude=-20, method='nearest').plot()
plt.title('Meridional speed across center (m/s)')
plt.savefig(f'figs/{vortex_type}_meridional_speed_across_center.png')
plt.close()



fig, axs = plt.subplots(1, 3, subplot_kw={
    'projection': ccrs.Orthographic(central_longitude=vortex_config['center'][0])}, figsize=[16, 4])
ftle_a.isel(time=0).plot(cmap=cmr.freeze, ax=axs[0], transform=ccrs.PlateCarree(), vmin=0)
ftle_r.isel(time=0).plot(cmap=cmr.flamingo, ax=axs[1], transform=ccrs.PlateCarree(), vmin=0)
(ftle_a.isel(time=0).where(ftle_a.isel(time=0)>0,0).drop('time') -
 ftle_r.where(ftle_r.isel(time=0)>0,0).isel(time=0).drop('time')).plot(
    levels=30, cmap=cmr.redshift_r, ax=axs[2], transform=ccrs.PlateCarree())
axs[0].set_title('Attracting FTLE')
axs[1].set_title('Repelling FTLE')
axs[2].set_title('Attracting - Repelling FTLE')
axs[0].coastlines(color='white')
axs[1].coastlines(color='white')
axs[2].coastlines(color='white')
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
axs[0].add_artist(draw_circle, )
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
axs[1].add_artist(draw_circle, )
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
axs[2].add_artist(draw_circle, )
plt.show()
plt.savefig(f'figs/FTLE_{vortex_type}.pdf')
plt.close()


## Dye plot

for t in range(x.time.values.shape[0]):
    u_plot = u.isel(time=t).values# - vortex_config['u_c']
    v_plot = v.isel(time=t).values# - vortex_config['v_c'] * np.sin(vortex_config['k']*2*np.pi*t/vortex_config['nt'])
    x_plot = v.isel(time=t).longitude.values
    y_plot = v.isel(time=t).latitude.values

    xx = x.isel(time=t).values.flatten()
    yy = y.isel(time=t).values.flatten()

    xxyy = zip(xx, yy)
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Orthographic(central_longitude=vortex_config['center'][0])},
                           )
    y_dye.isel(time=t).plot(cmap=cmr.iceburn, vmin=y.values.min(), vmax=y.values.max(),
                                 ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())
    ax.quiver(x_plot, y_plot, u_plot, v_plot, color='white',
                  transform=ccrs.PlateCarree())
    ax.scatter(xx, yy, alpha=0.1, color='white', transform=ccrs.PlateCarree(), edgecolor='white')

    ax.coastlines()
    ax.set_title(f'Mixing of a meridional dye - time {t}')
    draw_circle = plt.Circle(vortex_config['center'],
                             vortex_config['radius'], fill=False, color='red', transform=ccrs.PlateCarree())
    ax.add_artist(draw_circle)
    plt.savefig(f'figs/dye/{vortex_type}/'+'ideal_vortex_zonal_dye_global{:0>2}.png'.format( t), dpi=400)
    plt.close()


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
y_dye.isel(time=-1).plot.contourf(levels=20, cmap=cmr.iceburn, vmin=y.values.min(), vmax=y.values.max(), ax=ax, add_colorbar=True)
ax.streamplot(x=x_plot, y=y_plot, u=u_plot, v=v_plot)
ax.coastlines()
draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
ax.add_artist(draw_circle, )
plt.show()

for t in range(x.time.values.shape[0]):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
    ftle.isel(time=0).plot.contourf(levels=20, cmap=cmr.rainforest, vmin=0, ax=ax, vmax=0.15)
    draw_circle = plt.Circle(vortex_config['center'], vortex_config['radius'], fill=False, color='red')
    ax.coastlines()
    ax.add_artist(draw_circle)
    xx = x.isel(time=t).values.flatten()
    yy = y.isel(time=t).values.flatten()
    xxyy = zip(xx, yy)
    plt.scatter(xx, yy, alpha=0.5)
    plt.show()



for t in range(x.time.values.shape[0]):
    yy = y.isel(time=t).values.flatten()
    xx = x.isel(time=t).values.flatten()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 8])
    # ftle_r.isel(time=0).plot.contour(levels=20, cmap=cmr.flamingo, vmin=0, ax=ax, vmax=0.15)
    ax.hexbin(xx, yy, gridsize=30, vmin=0, vmax=10)
    ftle_a.isel(time=0).plot.contour(levels=20, cmap=cmr.freeze, vmin=0, ax=ax, vmax=0.15)
    xxyy = zip(xx, yy)
    ax.coastlines()
    plt.savefig(f'figs/hexbin/test_{t}_hexbin.png')
    plt.close()



