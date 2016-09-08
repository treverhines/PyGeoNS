#!/usr/bin/env python
import numpy as np
import pygeons.ioconv
import pygeons.dateconv
import rbf.halton

Nx = 200
start_time = pygeons.dateconv.decday('2000-01-01','%Y-%m-%d')
stop_time = pygeons.dateconv.decday('2001-01-01','%Y-%m-%d')
time = np.arange(start_time,stop_time+1)

Nt = len(time)
pos = 2*np.pi*(rbf.halton.halton(Nx,2) - 0.5)
lon = pos[:,0] - 84.5
lat = pos[:,1] + 43.0

east = 100*np.sin(lon[None,:])*np.cos(lat[None,:])*np.sin(time[:,None]/365.25)
north = 100*np.cos(lon[None,:])*np.sin(lat[None,:])*np.sin(time[:,None]/365.25)
vertical = 100*np.cos(lon[None,:])*np.cos(lat[None,:])*np.sin(time[:,None]/365.25)

east_std = 2.0*np.ones((Nt,Nx))
north_std = 2.0*np.ones((Nt,Nx))
vertical_std = 2.0*np.ones((Nt,Nx))
id = np.arange(Nx).astype(str)

data = {'time':time,
        'longitude':lon,
        'latitude':lat,
        'id':id,
        'east':east + np.random.normal(0.0,east_std),
        'north':north + np.random.normal(0.0,north_std),
        'vertical':vertical + np.random.normal(0.0,vertical_std),
        'east_std':east_std,
        'north_std':north_std,
        'vertical_std':vertical_std}

pygeons.ioconv.csv_from_dict('data/synthetic.csv',data)
