# This script generates the synthetic data file data.csv

import numpy as np
from pygeons.basemap import make_basemap
from pygeons.mjd import mjd
from pygeons.io.convert import text_from_dict
np.random.seed(1)

def make_data(pos,times):
  ''' 
  1 microstrain per year = 3 * 10^-9 strain per day
  '''  
  ms = 2.737e-9
  x,y = pos.T
  _,xg = np.meshgrid(times,x,indexing='ij')
  tg,yg = np.meshgrid(times,y,indexing='ij')
  u = 0.0*ms*tg*xg + 1.0*ms*tg*yg
  v = 1.0*ms*tg*xg + 0.0*ms*tg*yg
  z = 0.0*tg
  u = u - u[0,:]
  v = v - v[0,:]
  z = z - z[0,:]
  return u,v,z

lon = np.array([-83.3,-82.75,-85.26,-83.36])
lat = np.array([42.31,42.91,45.20,42.92])
id = np.array(['STA1','STA2','STA3','STA4'])
bm = make_basemap(lon,lat)
x,y = bm(lon,lat)
xy = np.array([x,y]).T
start_date = mjd('2000-01-01','%Y-%m-%d')
stop_date = mjd('2000-02-01','%Y-%m-%d')
times = np.arange(start_date,stop_date+1)
u,v,z = make_data(xy,times)
su = 0.001*np.ones_like(u)
sv = 0.001*np.ones_like(v)
sz = 0.001*np.ones_like(z)
u += np.random.normal(0.0,su)
v += np.random.normal(0.0,sv)
z += np.random.normal(0.0,sz)

data = {}
data['id'] = id
data['longitude'] = lon
data['latitude'] = lat
data['time'] = times
data['east'] = u
data['north'] = v
data['vertical'] = z
data['east_std'] = su
data['north_std'] = sv
data['vertical_std'] = sv
data['time_exponent'] = 0
data['space_exponent'] = 1
text_from_dict('data.csv',data)

