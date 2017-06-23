import numpy as np
import slippy.patch
import slippy.basis
import slippy.io
import slippy.gbuild
import slippy.bm
import slippy.quiver
from pygeons.mjd import mjd
from pygeons.io.io import text_from_dict
import matplotlib.pyplot as plt
np.random.seed(1)

### patch specifications
#####################################################################
strike = 70.0 # degrees
dip = 45.0 # degrees
length = 200000.0 # meters
width = 60000.0 # meters
seg_pos_geo = [-84.2,43.3,-50000.0] # top center of patch
Nl = 60
Nw = 30

## observation points
#####################################################################
Nx = 50
#pos_geo = np.random.uniform(-3,3,(Nx,3))
pos_geo = np.random.normal(0.0,0.5,(Nx,3))
pos_geo[:,0] += -84.2
pos_geo[:,1] += 43.3
pos_geo[:,2] = 0.0
disp_basis = slippy.basis.cardinal_basis((Nx,3))

# flatten the observation points and basis vectors
pos_geo_f = pos_geo[:,None,:].repeat(3,axis=1).reshape((Nx*3,3))
disp_basis_f = disp_basis.reshape((Nx*3,3))

### convert from cartesian to geodetic
bm = slippy.bm.create_default_basemap(pos_geo[:,0],pos_geo[:,1])
seg_pos_cart = slippy.bm.geodetic_to_cartesian(seg_pos_geo,bm)
pos_cart_f = slippy.bm.geodetic_to_cartesian(pos_geo_f,bm)
pos_cart = slippy.bm.geodetic_to_cartesian(pos_geo,bm)

### create synthetic slip
#####################################################################
P = slippy.patch.Patch(seg_pos_cart,length,width,strike,dip)
Ps = np.array(P.discretize(Nl,Nw))
Ns = len(Ps)
# find the centers of each patch in user coordinates
patch_pos = [P.user_to_patch(i.patch_to_user([0.5,0.5,0.0])) for i in Ps]
patch_pos = np.asarray(patch_pos)

# define slip as a function of patch position
slip = np.zeros((len(Ps),3))
slip[:,0] = 1.0/(1.0 + 20*((patch_pos[:,0] - 0.5)**2 + (patch_pos[:,1] - 0.5)**2))

slip_basis = slippy.basis.cardinal_basis((Ns,3))
slip_basis_f = slip_basis.reshape((Ns*3,3))

slip_f = slip.reshape((Ns*3,))

patches_f = Ps[:,None].repeat(3,axis=1).reshape((Ns*3,))

### create synthetic data
#####################################################################
G = slippy.gbuild.build_system_matrix(pos_cart_f,patches_f,disp_basis_f,slip_basis_f)
dx = 1.0
dy = 1.0
G_dx = slippy.gbuild.build_system_matrix(pos_cart_f + np.array([dx,0.0,0.0]),patches_f,disp_basis_f,slip_basis_f)
G_dy = slippy.gbuild.build_system_matrix(pos_cart_f + np.array([0.0,dy,0.0]),patches_f,disp_basis_f,slip_basis_f)

disp_f = G.dot(slip_f)
disp_dx_f = G_dx.dot(slip_f)
disp_dy_f = G_dy.dot(slip_f)

disp = disp_f.reshape((Nx,3))
disp_dx = disp_dx_f.reshape((Nx,3))
disp_dy = disp_dy_f.reshape((Nx,3))

u,v,z = disp.T
dudx = (disp_dx[:,0] - disp[:,0])/dx
dvdx = (disp_dx[:,1] - disp[:,1])/dx
dzdx = (disp_dx[:,2] - disp[:,2])/dx
dudy = (disp_dy[:,0] - disp[:,0])/dy
dvdy = (disp_dy[:,1] - disp[:,1])/dy
dzdy = (disp_dy[:,2] - disp[:,2])/dy

# make disp. time dependent
start_time = mjd('2016-01-01','%Y-%m-%d')
stop_time = mjd('2017-01-01','%Y-%m-%d')
peak_time = float(mjd('2016-07-01','%Y-%m-%d'))
times = np.arange(start_time,stop_time+1).astype(float)
Nt = len(times)
# slip rate (m/day) through time
b = 0.005/(((times-peak_time)/5.0)**2 + 1.0)  
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
su = 0.001*np.ones((Nt,Nx))
sv = 0.001*np.ones((Nt,Nx))
sz = 0.001*np.ones((Nt,Nx))
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
text_from_dict('soln.dx.csv',data)

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
text_from_dict('soln.dy.csv',data)
