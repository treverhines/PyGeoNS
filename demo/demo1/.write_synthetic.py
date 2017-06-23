import numpy as np
from pygeons.mjd import mjd
from pygeons.io.io import text_from_dict
from pygeons.basemap import make_basemap
import matplotlib.pyplot as plt
np.random.seed(1)

## observation points
#####################################################################
pos_geo = np.array([[-83.74,42.28,0.0],
                    [-83.08,42.33,0.0],
                    [-83.33,41.94,0.0]])  
Nx = len(pos_geo)
bm = make_basemap(pos_geo[:,0],pos_geo[:,1])
pos_cart = np.array(bm(pos_geo[:,0],pos_geo[:,1])).T
dx = pos_cart[:,0] - pos_cart[0,0]
dy = pos_cart[:,1] - pos_cart[0,1]

dispdx = np.array([[0.0,1e-6,0.0]]).repeat(Nx,axis=0)
dispdy = np.array([[0.0,0.0,0.0]]).repeat(Nx,axis=0)
disp = dispdx*dx[:,None] + dispdy*dy[:,None]
u,v,z = disp.T
dudx,dvdx,dzdx = dispdx.T
dudy,dvdy,dzdy = dispdy.T

# make disp. time dependent
start_time = mjd('2015-07-01','%Y-%m-%d')
stop_time = mjd('2017-07-01','%Y-%m-%d')
peak_time = float(mjd('2016-07-01','%Y-%m-%d'))
times = np.arange(start_time,stop_time+1).astype(float)
Nt = len(times)
# slip rate (m/day) through time
b = 0.005/(((times-peak_time)/10.0)**2 + 1.0)  
# slip (m) through time
intb = np.cumsum(b)

# create deformation rate gradients
#ddudx
# create displacements
u = u[None,:]*intb[:,None]
v = v[None,:]*intb[:,None]
z = z[None,:]*intb[:,None]

dudxdt = dudx[None,:]*b[:,None]
dvdxdt = dvdx[None,:]*b[:,None]
dzdxdt = dzdx[None,:]*b[:,None]

dudydt = dudy[None,:]*b[:,None]
dvdydt = dvdy[None,:]*b[:,None]
dzdydt = dzdy[None,:]*b[:,None]

# add noise
su = 0.0005*np.ones((Nt,Nx))
sv = 0.0005*np.ones((Nt,Nx))
sz = 0.0005*np.ones((Nt,Nx))
u += np.random.normal(0.0,su)
v += np.random.normal(0.0,sv)
z += np.random.normal(0.0,sz)

# time evolution
### write synthetic data
#####################################################################
data = {}
data['id'] = np.array(['A%03d' % i for i in range(Nx)])
data['longitude'] = pos_geo[:,0]
data['latitude'] = pos_geo[:,1]
data['time'] = times
data['east'] = u
data['north'] = v
data['vertical'] = z
data['east_std_dev'] = su
data['north_std_dev'] = sv
data['vertical_std_dev'] = sz
data['time_exponent'] = 0
data['space_exponent'] = 1
text_from_dict('data.csv',data)

# xder
data = {}
data['id'] = np.array(['A%03d' % i for i in range(Nx)])
data['longitude'] = pos_geo[:,0]
data['latitude'] = pos_geo[:,1]
data['time'] = times
data['east'] = dudxdt
data['north'] = dvdxdt
data['vertical'] = dzdxdt
data['east_std_dev'] = np.zeros((Nt,Nx))
data['north_std_dev'] = np.zeros((Nt,Nx))
data['vertical_std_dev'] = np.zeros((Nt,Nx))
data['time_exponent'] = -1
data['space_exponent'] = 0
text_from_dict('soln.dudx.csv',data)

# yder
data = {}
data['id'] = np.array(['A%03d' % i for i in range(Nx)])
data['longitude'] = pos_geo[:,0]
data['latitude'] = pos_geo[:,1]
data['time'] = times
data['east'] = dudydt
data['north'] = dvdydt
data['vertical'] = dzdydt
data['east_std_dev'] = np.zeros((Nt,Nx))
data['north_std_dev'] = np.zeros((Nt,Nx))
data['vertical_std_dev'] = np.zeros((Nt,Nx))
data['time_exponent'] = -1
data['space_exponent'] = 0
text_from_dict('soln.dudy.csv',data)
