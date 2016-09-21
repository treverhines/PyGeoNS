#!/usr/bin/env python
# generate synthetic data for an earthquake with afterslip
import numpy as np
import slippy.patch
import slippy.okada
import pygeons.ioconv
import pygeons.interface
import pygeons.dateconv
import matplotlib.pyplot as plt
np.random.seed(2)

## observation points
#####################################################################
Nx = 100
#pos_geo = np.random.uniform(-3,3,(Nx,3))
pos_geo = np.random.normal(0.0,1.0,(Nx,3))
pos_geo[:,0] += -84.2
pos_geo[:,1] += 43.3
pos_geo[:,2] = 0.0

# cartesian observation position
bm = pygeons.interface._make_basemap(pos_geo[:,0],pos_geo[:,1],resolution='i')
pos = np.array(bm(pos_geo[:,0],pos_geo[:,1])).T
pos = np.array([pos[:,0],pos[:,1],0*pos[:,0]]).T

### patch specifications
#####################################################################
strike = 0.0 # degrees
dip = 89.9 # degrees
length = 200000.0 # meters
width = 60000.0 # meters
seg_pos_geo = [-84.2,43.3,0.0] # top center of patch
slip = [1.0,0.0,0.0]
Nl = 60
Nw = 30

# cartesian segment position
seg_pos = np.array(bm(seg_pos_geo[0],seg_pos_geo[1]))
seg_pos = np.array([seg_pos[0],seg_pos[1],0*seg_pos[0]])
### create patch instance
#####################################################################
p = slippy.patch.Patch(seg_pos,length,width,strike,dip)
# print patch trace endpoints in lon,lat
point1 = bm(*p.patch_to_user([0.0,1.0,0.0])[:2],inverse=True)
point2 = bm(*p.patch_to_user([1.0,1.0,0.0])[:2],inverse=True)
print('endpoint 1 : lon %s lat %s' % point1)
print('endpoint 2 : lon %s lat %s' % point2)

### compute displacements greens functions
#####################################################################
disp,derr = slippy.okada.patch_dislocation(pos,slip,p)

# XXXXXXXXXXXXXXXXXXXXX
disp[:,0] = 0.0
disp[:,1] = pos[:,1]#np.arctan((pos[:,0]-seg_pos[0])/100000.0)
disp[:,2] = 0.0
# XXXXXXXXXXXXXXXXXXXXX

time_start = pygeons.dateconv.decday('2000-01-01','%Y-%m-%d')
time_stop = pygeons.dateconv.decday('2002-01-01','%Y-%m-%d')
time_eq = pygeons.dateconv.decday('2001-01-01','%Y-%m-%d')
times = np.arange(int(time_start),int(time_stop))
Nt = len(times)

data = np.zeros((Nt,Nx,3))
slip_history = np.zeros(Nt)
#slip_history[times>=time_eq] = 5.0 - np.exp(-(times[times>=time_eq]-time_eq)/100.0)
# XXXXXXXXXXXXXXXXXXXXX
slip_history[...] = 1.0
# XXXXXXXXXXXXXXXXXXXXX

data = slip_history[:,None,None]*disp[None,:]
sigma = 0.01*np.ones(data.shape)
data += np.random.normal(0.0,sigma)

data_dict = {}
data_dict['id'] = np.arange(Nx).astype(str)
data_dict['longitude'] = pos_geo[:,0]
data_dict['latitude'] = pos_geo[:,1]
data_dict['time'] = times
data_dict['east'] = data[:,:,0]
data_dict['north'] = data[:,:,1]
data_dict['vertical'] = data[:,:,2]
data_dict['east_std'] = sigma[:,:,0]
data_dict['north_std'] = sigma[:,:,1]
data_dict['vertical_std'] = sigma[:,:,2]
data_dict['time_power'] = 0
data_dict['space_power'] = 1
pygeons.ioconv.csv_from_dict('data/synthetic.csv',data_dict)
quit()
print(data.shape)


