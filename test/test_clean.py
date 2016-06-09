#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt

Nt = 3000
Nx = 3
t = np.linspace(0.0,5.0,Nt)
x = np.zeros((Nx,2))
x = np.random.random((Nx,2))
u = np.exp(-t)[:,None]
u = np.arctan(t-2.0)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,0.1,(Nt,Nx))
sigma = 0.1*np.ones((Nt,Nx))
#u[45] = 0.3
#u[55] = 0.8

outliers = pygeons.clean.outliers(u,t,x,sigma=sigma,tol=3.0,alpha=None)

fig,ax = plt.subplots()
ax.plot(t,u,'.')
plt.show()
print(u.shape)


