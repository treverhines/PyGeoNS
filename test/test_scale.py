#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pygeons.smooth
import pygeons.diff

def autocorr(x):
    x = np.asarray(x)
    n = x.shape[0]
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


# signal length scale
x_scale = 1.0
# uncertainty scale
y_scale = 0.1

p = (x_scale/4.0)**2/y_scale
sig = 0.1
N = 1000
tests = 50
time = np.linspace(0.0,10.0,N)
x = np.array([[0.0,0.0]])

data = np.random.normal(0.0,sig,(N,1))
data += np.sin(time)[:,None]

sigma = sig*np.ones((N,1))
out = pygeons.smooth.network_smoother(data,time,x,sigma=sigma,diff_specs=[pygeons.diff.ACCELERATION],penalties=[p],procs=0)
results = out[0][:,0]

fig,ax = plt.subplots()
ax.plot(time,data,'k.')
ax.plot(time,results,'b-')
plt.show()

