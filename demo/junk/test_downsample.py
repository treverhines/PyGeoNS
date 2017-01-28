#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pygeons.smooth import network_smoother
from pygeons.downsample import MeanInterpolant
from pygeons.downsample import network_downsampler
import pygeons.diff
import pygeons.cuts
import modest
  
N1 = 10000
N2 = 20
T = 0.1

x = np.array([[0.0,0.0]])
t1 = np.linspace(0.0,1.0,N1)
t2 = np.linspace(0.0,1.0,N2)
dt1 = t1[1] - t1[0]
dt2 = t2[1] - t2[0]
print('dt1 : %s' % dt1)
print('dt2 : %s' % dt2)

sigma_scale = 2.0
sigma1 = sigma_scale*np.ones((N1,1))
u1 = 10*np.sin(10*t1)[:,None] + np.random.normal(0.0,sigma1)
u2,sigma2 = network_downsampler(u1,t1,t2,x,sigma=sigma1)
fig,ax = plt.subplots()
ax.plot(t1,u1,'k.')
ax.plot(t2,u2,'b.')

smooth1,perts1 = network_smoother(u1,t1,x,sigma=sigma1,
                                 diff_specs=[pygeons.diff.acc()],
                                 time_scale=T,perts=500)
std1 = np.std(perts1,axis=0)
print(std1.shape)
smooth2,perts2 = network_smoother(u2,t2,x,sigma=sigma2,
                                 diff_specs=[pygeons.diff.acc()],
                                 time_scale=T,perts=500)

std2 = np.std(perts2,axis=0)

fig,ax = plt.subplots()
ax.plot(t1,smooth1,'r-')
ax.fill_between(t1,(smooth1-std1)[:,0],(smooth1+std1)[:,0],color='r',alpha=0.5)
ax.plot(t2,smooth2,'b-')
ax.fill_between(t2,(smooth2-std2)[:,0],(smooth2+std2)[:,0],color='b',alpha=0.5)
plt.show()
