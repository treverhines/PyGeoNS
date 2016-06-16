#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt
import logging
import pygeons.diff
logging.basicConfig(level=logging.INFO)

Nt = 10000
Nx = 1
S = 0.1
t = np.linspace(0.0,5.0,Nt)
dt = 5.0/Nt
x = np.zeros((Nx,2))
x = np.random.random((Nx,2))
u = 1*np.sin(2*t)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,S,(Nt,Nx))
u += np.random.normal(0.0,S,(Nt,1))
sigma = S*np.ones((Nt,Nx))

fig,ax = plt.subplots()
ax.plot(t,u,'ro',zorder=1)
u,sigma = pygeons.clean.network_cleaner(u,t,sigma=sigma,
                                        time_scale = 10*dt,
                                        plot=True,
                                        use_umfpack=True)
ax.plot(t,u,'bo',zorder=1)
plt.show()
quit()


