#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt
import logging
import pygeons.diff
import pygeons.smooth
logging.basicConfig(level=logging.INFO)

Nt = 100
Nx = 1
S = 0.1
t = np.linspace(0.0,5.0,Nt)
x = np.random.random((Nx,2))
u = 1*np.sin(2*t)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,S,(Nt,Nx))


mask = np.zeros((Nt,Nx),dtype=bool)
mask[(t>2) & (t<4.5)] = True

sigma = S*np.ones((Nt,Nx))
sigma[mask] = np.inf
u[mask] = np.nan

#tc = pygeons.cuts.TimeCuts([1.6])
ds = pygeons.diff.acc()
#ds['time']['cuts'] = tc
#sigma[t > 1.5] = np.inf
us = pygeons.smooth.smooth(t,x,u,sigma=sigma,diff_specs=[ds],fill='interpolate')
print(us)
plt.plot(t,u[:,0],'ko')
plt.plot(t,us[:,0],'b-')
plt.show()

