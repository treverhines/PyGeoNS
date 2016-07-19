#!/usr/bin/env python
import numpy as np
import pygeons.diff
import pygeons.smooth
import pygeons.cuts
import pygeons.quiver
import matplotlib.pyplot as plt
import logging
import rbf.halton
import rbf.basis
import pygeons.view
rbf.basis.set_sym_to_num('numpy')
import scipy.sparse
logging.basicConfig(level=logging.DEBUG)

  
t = np.linspace(0.0,1.0,3)
#x = np.array([[0.0,0.0]])
x = rbf.halton.halton(200,2)
u = np.sin(5*t[:,None])*np.cos(5*x[:,0])[None,:]
u += np.random.normal(0.0,0.1,u.shape)
sigma = np.ones(u.shape)
#sigma[(t>0.5)&(t<0.8)] = np.inf
#u[(t>0.5)&(t<0.8)] = 1.0
#u[(t<0.5)] += 1.0


#for i in range(1000):
#  r1 = np.random.randint(0,1000)
#  r2 = np.random.randint(0,100)
#  sigma[r1,r2] = np.inf
us1 = pygeons.smooth.smooth(t,x,u,sigma=sigma,time_scale=0.05,fill=True)
us2 = pygeons.smooth.smooth(t,x,u,sigma=sigma,time_scale=0.05,fill=False)
pygeons.view.view(t,x,z=[u,us1,us2])

print(us1.shape)
plt.plot(t,u,'k-')
plt.plot(t,us1,'b-')
plt.plot(t,us2,'r-')
plt.show()
#u += np.random.normal(0.0,0.1,u.shape)
#sigma = np.ones(u.shape)
#sigma[10,2] = np.inf
#sigma[30:,0] = np.inf

#acc['space']['stencil_size'] = 2
#acc['time']['cuts'] = cuts

#mask = np.isinf(sigma)
#pygeons.diff.diff_matrix(t,x,acc,mask=mask)


