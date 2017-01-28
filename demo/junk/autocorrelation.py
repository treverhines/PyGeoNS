#!/usr/bin/env python
import numpy as np
import pygeons.smooth
import matplotlib.pyplot as plt
import pygeons.diff

N = 100
t = np.linspace(0.0,1.0,N)
u = 100*np.random.normal(0.0,1.0,N)
x = np.array([[0.0,0.0]])
out = pygeons.smooth.network_smoother(u[:,None],t,x,perts=100,
       diff_specs=[pygeons.diff.ACCELERATION],penalties=[0.0001])
us = out[0][:,0]
pert = out[1][:,:,0]

corr = np.correlate(us,us,'full')
corr /= np.max(corr)
corr2 = np.corrcoef(pert.T)
print(corr2.shape)
quit()
fig,ax = plt.subplots()
ax.plot(us)
fig,ax = plt.subplots()
ax.plot(corr)
plt.show()

