#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import rbf.halton
import rbf.basis
import modest
import gps.plot
from pygeons.clean import interactive_cleaner

t = np.linspace(0,1,100) # form observation times
x = np.random.normal(0.0,1.0,(20,2)) # form observation positions
x[:,0] += -84.0
x[:,1] += 43.0

fig,ax = plt.subplots()
bm = gps.plot.create_default_basemap(x[:,1],x[:,0])
bm.drawstates(ax=ax)
bm.drawcountries(ax=ax)
bm.drawcoastlines(ax=ax)
bm.drawparallels(np.arange(30,90),ax=ax)
bm.drawmeridians(np.arange(-100,-60),ax=ax)
#help(bm.drawmeridians)
pos_x,pos_y = bm(x[:,0],x[:,1])
pos = np.array([pos_x,pos_y]).T
x = pos

u,v,z = np.random.normal(0.0,0.5,(3,100,20))
su = 0.5 + 0*u
sv = 0.5 + 0*u
sz = 0.5 + 0*u

sigma_set = np.zeros((100,20,3))
data_set = np.zeros((100,20,3))
data_set[:,:,0] = u
data_set[:,:,1] = v
data_set[:,:,2] = z
sigma_set[:,:,0] = su
sigma_set[:,:,1] = sv
sigma_set[:,:,2] = sz

data_set[60,10,:] += 10

data_set[50:,:,0] += 5.0
data_set[50:,:,1] += 10.0
data_set[50:,:,2] += 7.0

interactive_cleaner(t,x,
                        data_set[:,:,0],data_set[:,:,1],data_set[:,:,2],
                        sigma_set[:,:,0],sigma_set[:,:,1],sigma_set[:,:,2])
#network_viewer(data_set,t,x,u=[u1,u2],v=[v1,v2],z=[z1,z2],su=[su,su],sv=[sv,sv],
#               map_ax=ax) 
#network_viewer(t,x,u=[u1],v=[v1],z=[z1],su=[su],sv=[sv],sz=[sz],
#               map_ax=ax) 
#network_viewer(t,pos,u=[u1,u2],v=[v1,v2],z=[z1,z2],map_ax=ax,quiver_scale=0.00001) 
