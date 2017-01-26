# This script generates synthetic data along the Cascadia subduction 
# zone.  The east, north, and vertical component of the synthetic data 
# are
#
#   u(x,y) = 0.01*cos(x*w*2*pi)*sin(y*w*2*pi) + eps
#   v(x,y) = 0.01*sin(x*w*2*pi)*cos(y*w*2*pi) + eps
#   z(x,y) = 0.01*cos(x*w*2*pi)*cos(y*w*2*pi) + eps
#
# where w is the cutoff frequency (1.0/400000 km^-1), and eps is white 
# noise with standard deviation 0.005 m.


import numpy as np
import matplotlib.pyplot as plt
import pygeons.basemap
import pygeons.mjd
import pygeons.io

def make_data(pos,times):
  '''returns synthetic displacements'''
  Nx = len(pos)
  Nt = len(times)
  u = np.zeros((Nt,Nx))
  v = np.zeros((Nt,Nx))
  z = np.zeros((Nt,Nx))
  u[...] = 0.001*pos[:,0] + 0.004*pos[:,1] 
  v[...] = 0.002*pos[:,0] + 0.005*pos[:,1] 
  z[...] = 0.003*pos[:,0] + 0.006*pos[:,1] 
  mean = np.concatenate((u[...,None],v[...,None],z[...,None]),axis=2)
  return mean

# load Cascadia GPS station location
lonlat = np.loadtxt('.make_data/lonlat.txt')
pos_geo = np.zeros((lonlat.shape[0],3))
pos_geo[:,0] = lonlat[:,0]
pos_geo[:,1] = lonlat[:,1]

# convert geodetic coordinates to cartesian
bm = pygeons.basemap.make_basemap(pos_geo[:,0],pos_geo[:,1],resolution='i')
pos = np.array(bm(pos_geo[:,0],pos_geo[:,1])).T
pos = np.array([pos[:,0],pos[:,1],0*pos[:,0]]).T

time_start = pygeons.mjd.mjd('2000-01-01','%Y-%m-%d')
time_stop = pygeons.mjd.mjd('2000-01-02','%Y-%m-%d')
times = np.arange(int(time_start),int(time_stop))

mean = make_data(pos,times)
sigma = 0.00001*np.ones(mean.shape)
data_dict = {}
data_dict['id'] = np.arange(len(pos)).astype(str)
data_dict['longitude'] = pos_geo[:,0]
data_dict['latitude'] = pos_geo[:,1]
data_dict['time'] = times
data_dict['east'] = mean[:,:,0]
data_dict['north'] = mean[:,:,1]
data_dict['vertical'] = mean[:,:,2]
data_dict['east_std'] = sigma[:,:,0]
data_dict['north_std'] = sigma[:,:,1]
data_dict['vertical_std'] = sigma[:,:,2]
data_dict['time_exponent'] = 0
data_dict['space_exponent'] = 1
pygeons.io.convert.text_from_dict('synthetic.nonoise.csv',data_dict)

mean = 0*make_data(pos,times)
sigma = 0.005*np.ones(mean.shape)
mean += np.random.normal(0.0,sigma)
data_dict = {}
data_dict['id'] = np.arange(len(pos)).astype(str)
data_dict['longitude'] = pos_geo[:,0]
data_dict['latitude'] = pos_geo[:,1]
data_dict['time'] = times
data_dict['east'] = mean[:,:,0]
data_dict['north'] = mean[:,:,1]
data_dict['vertical'] = mean[:,:,2]
data_dict['east_std'] = sigma[:,:,0]
data_dict['north_std'] = sigma[:,:,1]
data_dict['vertical_std'] = sigma[:,:,2]
data_dict['time_exponent'] = 0
data_dict['space_exponent'] = 1
pygeons.io.convert.text_from_dict('synthetic.csv',data_dict)

