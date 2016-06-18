#!/usr/bin/env python
import numpy as np
import pygeons.clean
import matplotlib.pyplot as plt
import logging
import pygeons.diff
logging.basicConfig(level=logging.INFO)

Nt = 1000
Nx = 10
S = 0.1
t = np.linspace(0.0,5.0,Nt)
#t = np.random.random((Nt,))
dt = 1.0
x = np.zeros((Nx,2))
x = np.random.random((Nx,2))
u = 1*np.sin(2*t)[:,None]
u = u.repeat(Nx,axis=1)
u += np.random.normal(0.0,S,(Nt,Nx))
u += np.random.normal(0.0,2*S,(Nt,1))
#u[5000,0] = 100
sigma = S*np.ones((Nt,Nx))
#sigma[t<=2.0,1] = np.inf
un,sigman = pygeons.clean.network_cleaner(u,t,x,sigma=sigma,plot=False,zero_idx=0)
#u_out,sigma_out = pygeons.smooth.network_smoother(u,t,x,sigma=sigma,perts=100)
#ubl,sigmabl = pygeons.clean.baseline(u,t,x,sigma=sigma,perts=100,zero_idx=0)

#sigma_out = np.std(sigma_out,axis=0)
#print(u)
#ax.plot(t,u,'ro',zorder=1)
fig,ax = plt.subplots()
ax.errorbar(t,u[:,0],sigma[:,0],color='k')
ax.errorbar(t,un[:,0],sigman[:,0],color='b')
ax.plot(t,np.sin(2*t),'k-')
plt.show()
quit()


