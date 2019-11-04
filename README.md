# lagrangian-coherence
Library for computing the Finite-time Lyapunov Exponents (FTLE) of 2D flows using xarray. 
The main purpose of this LagrangianCoherence is to ease the application of Lyapunov exponents
in atmospheric data, usually stored in netcdf files. Xarray provides an interface to ncdf files as 
well as computational capabilities. We employed numba to accelerate the computationally expensive
tasks.

Generating some sample atmospheric flow:
        
    lat1, lat2 = -80, 20
    lon1, lon2 = -80, -30
    dx = 0.5
    earth_r = 6371000
    timestep = 6 * 3600
    ntime = 4
    ky, kx = 10, 40
        frq = 0.25

    time_dir = 'backward'
    timestep = -timestep if time_dir == 'backward' else timestep

    nlat = int((lat2 - lat1) / dx)
    nlon = int((lon2 - lon1) / dx)
    latitude = np.linspace(lat1, lat2, nlat)
    longitude = np.linspace(lon1, lon2, nlon)
    time = pd.date_range("2000-01-01T00:00:00", periods=ntime, freq="6H")
    time_idx = np.array([x for x in range(len(time))])
    u_data = 20 * np.ones([nlat, nlon, len(time)]) * (np.sin(ky*np.pi*latitude/180)**2).reshape([nlat,1,1]) * np.cos(time_idx * frq).reshape([1,1,len(time)])
    v_data = 40 * np.ones([nlat, nlon, len(time)]) * (np.sin(kx*np.pi*longitude/360)**2).reshape([1,nlon,1]) * np.cos(time_idx * frq).reshape([1,1,len(time)])

    u = xr.DataArray(u_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude, 'longitude': longitude, 'time': time})
    v = xr.DataArray(v_data, dims=['latitude', 'longitude', 'time'],
                     coords={'latitude': latitude, 'longitude': longitude, 'time': time})
    u.name = 'u'
    v.name = 'v'
    ds = xr.merge([u, v])

Computing the FTLE:

    lcs = LCS(lcs_type='repelling', timestep=timestep, timedim='time')

    ftle = lcs(ds)

