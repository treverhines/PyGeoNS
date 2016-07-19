#!/usr/bin/env python
import numpy as np
import rbf.halton
import pygeons.interface
import pygeons.ioconv
import pygeons.decyear
import logging
logging.basicConfig(level=logging.INFO)

Nx = 50
time = pygeons.decyear.decyear_range('2000-01-01','2010-01-05',1.5,'%Y-%m-%d')
Nt = len(time)
lon = np.random.normal(-84.5,2.0,Nx)
lat = np.random.normal(43.0,2.0,Nx)
u = 2*(time[:,None]-2000) + np.random.normal(0.0,1.0,(Nt,Nx))
v = 3*(time[:,None]-2000) + np.random.normal(0.0,1.0,(Nt,Nx))
z = 4*np.sin(lon[None,:])*np.cos(time[:,None]-2000) + np.random.normal(0.0,1.0,(Nt,Nx))
su = np.ones((Nt,Nx))
sv = np.ones((Nt,Nx))
sz = np.ones((Nt,Nx))

data = {}
data['id'] = np.arange(Nx).astype(str)
data['time'] = time
data['longitude'] = lon
data['latitude'] = lat
data['east'] = u
data['north'] = v
data['vertical'] = z
data['east_std'] = su
data['north_std'] = sv
data['vertical_std'] = sz

#data = pygeons.interface.clean(data)
data = pygeons.interface.downsample(data,50,'2005-01-01','2010-01-01')
pygeons.ioconv.file_from_dict('test.csv',data)
data1 = pygeons.ioconv.dict_from_file('test.csv')
data2 = pygeons.interface.smooth(data,time_scale=1.0,length_scale=100000)
data3 = pygeons.interface.diff(data2,dt=1,dx=1)
pygeons.interface.check_compatability([data,data1,data2,data3])
#data_diff = pygeons.interface.diff(data,dt=1)
#pygeons.ioconv.file_from_dict('test_data.csv',data)
#data2 = pygeons.ioconv.dict_from_file('test_data.csv')

pygeons.interface.view([data1,data2,data3],data_set_names=['observed','smoothed','diffed'])
