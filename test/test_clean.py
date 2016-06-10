#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
Nt = 1000
Nx = 100
t = np.linspace(0.0,5.0,Nt)
x = np.zeros((Nx,2))
x = np.random.random((Nx,2))
u = np.sin(t)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,0.1,(Nt,Nx))
u += np.random.normal(0.0,0.1,(Nt,1))
sigma =np.ones((Nt,Nx))


fig,ax = plt.subplots()
ax.plot(t,u,'ro',zorder=0)
u,sigma = pygeons.clean.network_cleaner(u,t,x,sigma=sigma,penalty=0.001,plot=False)
fig,ax = plt.subplots()
ax.plot(t,u,'bo',zorder=1)
plt.show()
quit()


