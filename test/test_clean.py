#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt
import logging
import pygeons.diff
logging.basicConfig(level=logging.INFO)

Nt = 1000
Nx = 1
S = 0.1
t = np.linspace(0.0,5.0,Nt)
x = np.zeros((Nx,2))
x = np.random.random((Nx,2))
u = 10*np.sin(t)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,S,(Nt,Nx))
u += np.random.normal(0.0,S,(Nt,1))
sigma = S*np.ones((Nt,Nx))

fig,ax = plt.subplots()
ax.plot(t,u,'ro',zorder=1)
u,sigma = pygeons.clean.network_cleaner(u,t,x,sigma=sigma,
                                        penalty=None,plot=True,cv_plot=True,
                                        use_umfpack=True,cv_bounds=[[-3.0,1.0]])
ax.plot(t,u,'bo',zorder=1)
plt.show()
quit()


