#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pygeons.smooth
import pygeons.diff
import rbf.halton
import rbf.nodes
import rbf.integrate
import logging
import myplot.cm
import scipy.signal
logging.basicConfig(level=logging.DEBUG)
np.random.seed(2)

''' 
## SCALING FOR TIME SMOOTHING 
#####################################################################
# cutoff frequency
cutoff = 1.0/2.0

# uncertainty
S = 1.0

# number of observations (should be odd so that it is centered on zero)
N = 5001

# number of perturbations to use when computing correlation. Should be 
# about 10000 for a good plots
PERTS = 10000

# observation points
t = np.linspace(-10,10,N)
t = np.random.uniform(-10,10,(N,))
t = np.sort(t)
x = np.array([[0.0,0.0]])

# generate synthetic data
data = np.random.normal(0.0,S,(N,1))
sigma = S*np.ones((N,1))

ds = pygeons.diff.acc()
ds['time']['diffs'] = [[2]]
smooth = pygeons.smooth.smooth(
           t,x,data,ds,
           sigma=sigma,
           time_cutoff=cutoff)

data_pert = np.random.normal(0.0,S,(PERTS,N,1))
data_pert += data
perts = pygeons.smooth.smooth(t,x,data_pert,ds,
                              sigma=sigma,
                              time_cutoff=cutoff)
# remove x axis
smooth = smooth[:,0]
perts = perts[:,:,0]

fig,ax = plt.subplots()
ax.plot(t,data[:,0],'k.')
ax.plot(t,smooth,'b-')

# compute correlation matrix. This should be the covariance matrix
C = np.corrcoef(perts.T)
corr = C[N//2,:]
fig,ax = plt.subplots()
ax.plot(t,corr,'-',lw=2)
ax.plot(t,np.sinc(t*cutoff*2.0),'r-')
plt.show()
'''
## SCALING FOR SPACE SMOOTHING 
#####################################################################
# spatial cutoff frequency
cutoff = 1.0/5.0

# uncertainty scale
S = 1.0

# number of observations 
N = 5000

# number of perturbations to use when computing correlation. Should be 
# about 1000 for a good plots
PERTS = 10000

# observation points
# define bounding circle
t = np.linspace(0.0,2*np.pi,100)
vert = 10*np.array([np.sin(t),np.cos(t)]).T
smp = np.array([range(100),np.roll(range(100),-1)]).T
fix = np.array([[0.0,0.0]])
x,sid = rbf.nodes.make_nodes(N-1,vert,smp,fix_nodes=fix)
x = np.vstack((fix,x))
t = np.array([0.0])

sigma = S*np.ones((1,N))
ds = pygeons.diff.disp_laplacian()
ds['space']['diffs'] = [[2,0],[0,2]]
data = np.random.normal(0.0,sigma)
smooth = pygeons.smooth.smooth(t,x,data,ds,
                               sigma=sigma,
                               space_cutoff=cutoff)
perts = np.random.normal(0.0,S,(PERTS,1,N))
perts = pygeons.smooth.smooth(t,x,perts,ds,
                              sigma=sigma,
                              space_cutoff=cutoff)
# remove x axis
smooth = smooth[0,:]
perts = perts[:,0,:]
# compute correlation matrix
C = np.corrcoef(perts.T)
# extract correlation for just x=[0.0,0.0]
corr = C[0,:]
# plot the data
fig,ax = plt.subplots()
c = ax.scatter(x[:,0],x[:,1],s=100,c=data[0,:],cmap='viridis',
               edgecolor='none')
cbar = plt.colorbar(c,ax=ax)
ax.grid()
fig,ax = plt.subplots()
c = ax.scatter(x[:,0],x[:,1],s=100,c=smooth,cmap='viridis',
               edgecolor='none')
cbar = plt.colorbar(c,ax=ax)
ax.grid()
fig,ax = plt.subplots()
c = ax.tripcolor(x[:,0],x[:,1],corr,cmap='seismic',
                 vmin=-1.0,vmax=1.0)
cbar = plt.colorbar(c,ax=ax)
ax.grid()
plt.show()

quit()
