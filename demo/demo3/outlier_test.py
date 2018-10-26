import numpy as np
import h5py
import matplotlib.pyplot as plt
import rbf
import pygeons
from matplotlib import colors
import logging
logging.basicConfig(level=logging.DEBUG)

c1 = colors.to_rgb('C0')
c1t = tuple(0.6 + 0.4*np.array(colors.to_rgb('C0')))
c2 = colors.to_rgb('C1')
c2t = tuple(0.6 + 0.4*np.array(colors.to_rgb('C1')))

dat = h5py.File('work/data.h5','r')

xidx = np.nonzero(dat['id'][...] == 'SC03')[0][0]
tidx = ((dat['time'][:] > pygeons.mjd.mjd('2015-08-01','%Y-%m-%d')) & 
        (dat['time'][:] < pygeons.mjd.mjd('2016-08-01','%Y-%m-%d')))

u = dat['east'][tidx,:][:,xidx]
s = dat['east_std_dev'][tidx,:][:,xidx]
t = dat['time'][tidx] 

mask = ~np.isinf(s)
u = 1000*u[mask]
s = 1000*s[mask]
t = t[mask]/365.25

def basis(t):
  return np.array([np.sin(t[:,0]),np.cos(t[:,0]),np.sin(2*t[:,0]),np.cos(2*t[:,0])]).T
  
# detect outliers
#p = np.array([t**0,t**1,
#              np.sin(t/365.25),np.cos(t/365.25),
#              np.sin(2*t/365.25),np.cos(2*t/365.25)]).T
#p = np.array([t**0,t**1]).T
#out_idx = rbf.gauss.outliers(u,s,p=p,tol=4.0)

gp = rbf.gauss.gppoly(1)
#gp += rbf.gauss.gpse((0.0,1.0,0.02))
gp += rbf.gauss.gpbfci(basis)

out_idx = gp.outliers(t[:,None],u,s,tol=3.0)


plt.errorbar(t,u,s,marker='.',linestyle='None',color=c1,ecolor=c1t)
plt.errorbar(t[out_idx],u[out_idx],s[out_idx],marker='.',linestyle='None',color=c2,ecolor=c2t)

plt.show()
