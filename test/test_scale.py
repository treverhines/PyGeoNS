#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pygeons.smooth
import pygeons.diff
import rbf.halton
import rbf.nodegen
import rbf.integrate
import logging
import myplot.cm
import scipy.signal
logging.basicConfig(level=logging.INFO)
np.random.seed(1)

## SCALING FOR TIME SMOOTHING 
#####################################################################
# length scale
T = 5.0
# uncertainty scale
S = 0.1

# damping parameters
Pscale = [0.05,0.25,1.0]
P = [i*T**2/S for i in Pscale]

# number of observations (should be odd so that it is centered on 
# zero)
N = 201

# number of perturbations to use when computing correlation. Should be 
# about 10000 for a good plots
PERTS = 10000

# observation points
t = np.linspace(-10*T,10*T,N)
x = np.array([[0.0,0.0]])

# generate synthetic data
data = np.random.normal(0.0,S,(N,1))
sigma = S*np.ones((N,1))

fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()
ax2.plot(t/T,data/S,'k.')
for i in range(len(Pscale)): 
  smooth,perts = pygeons.smooth.network_smoother(
                   data,t,x,sigma=sigma,
                   diff_specs=[pygeons.diff.ACCELERATION],
                   penalties=[P[i]],perts=PERTS,procs=6)
  # remove x axis
  smooth = smooth[:,0]
  perts = perts[:,:,0]
  # compute correlation matrix
  C = np.corrcoef(perts.T)
  # extract correlation for just t=0.0
  corr = C[N//2,:]

  ax1.plot(t/T,corr,'-',lw=2)
  ax2.plot(t/T,smooth/S,'-',lw=2)
  # sample spacing
  dt = (t[1] - t[0])/T
  freq,pow = scipy.signal.periodogram(corr,1.0/dt)
  ax3.loglog(1.0/freq,pow)
  #ax.set_title(u'penalty: %s (T^2/S)' % Pscale)

ax1.set_xlabel('time (T)')
ax1.set_ylabel('correlation')
ax1.legend(['penalty=%s' % i for i in Pscale],frameon=False)
ax1.set_ylim((-0.1,1.0))
ax1.set_xlim((-5.0,5.0))
ax1.grid()

#ax.legend(['observed','smoothed'],frameon=False)
ax2.legend(['observed'] + ['penalty=%s' % i for i in Pscale],frameon=False)
ax2.set_ylabel('displacement (S)')
ax2.set_xlabel('time (T)')
ax2.set_ylim((-4.0,4.0))
ax2.grid()
plt.show()
quit()
## SCALING FOR SPACE SMOOTHING 
#####################################################################

# length scale
L = 100.0
# uncertainty scale
S = 10.0

# damping parameters
Pscale = 0.25
P = Pscale*L**2/S

# number of observations 
N = 400

# number of perturbations to use when computing correlation. Should be 
# about 1000 for a good plots
PERTS = 5000

# observation points
# define bounding circle
t = np.linspace(0.0,2*np.pi,100)
vert = 5*L*np.array([np.sin(t),np.cos(t)]).T
smp = np.array([range(100),np.roll(range(100),-1)]).T
fix = np.array([[0.0,0.0]])
x,sid = rbf.nodegen.volume(N-1,vert,smp,fix_nodes=fix,n=5,itr=1000,delta=0.01)
x = np.vstack((fix,x))
t = np.array([0.0])

# make data
data = np.random.normal(0.0,S,(1,N))
sigma = S*np.ones((1,N))

smooth,perts = pygeons.smooth.network_smoother(
                   data,t,x,sigma=sigma,
                   diff_specs=[pygeons.diff.DISPLACEMENT_LAPLACIAN],
                   penalties=[P],perts=PERTS,procs=6)
# remove x axis
smooth = smooth[0,:]
perts = perts[:,0,:]
# compute correlation matrix
C = np.corrcoef(perts.T)
# extract correlation for just x=[0.0,0.0]
corr = C[0,:]

# plot the data
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=data[0,:]/S,cmap=myplot.cm.viridis,
      vmin=-2.0,vmax=2.0)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement (S)')

# plot smoothed data
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=smooth/S,cmap=myplot.cm.viridis,zorder=1,
               vmin=-0.7,vmax=0.7)
ax.tripcolor(x[:,0]/L,x[:,1]/L,smooth/S,cmap=myplot.cm.viridis,zorder=0,
               vmin=-0.7,vmax=0.7)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement (S)')

# plot correlation
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=corr,cmap='seismic',zorder=1,
               vmin=-1.0,vmax=1.0)
ax.tripcolor(x[:,0]/L,x[:,1]/L,corr,cmap='seismic',zorder=0,
               vmin=-1.0,vmax=1.0)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('correlation')
plt.show()

## SCALING FOR SPACE VELOCITY SMOOTHING 
#####################################################################

# time steps
Nt = 50

# total time 
T = 1000.0

# length scale
L = 1.0

# uncertainty scale
S = 1.0

# damping parameters. scale the damping parameter by the signal length 
# scale squared and the time step size. multiplying by the time step 
# size effectively means that we are putting a laplacian damping on 
# the difference between successive displacement observations, 
# which renders time irrelevant 
Pscale = 0.25
P = Pscale*L**2*(T/(Nt-1))/S

# number of observations 
N = 400

# number of perturbations to use when computing correlation. Should be 
# about 1000 for a good plots
PERTS = 100

# observation points
# define bounding circle
t = np.linspace(0.0,2*np.pi,100)
vert = 5*L*np.array([np.sin(t),np.cos(t)]).T
smp = np.array([range(100),np.roll(range(100),-1)]).T
fix = np.array([[0.0,0.0]])
x,sid = rbf.nodegen.volume(N-1,vert,smp,fix_nodes=fix,n=5,itr=1000,delta=0.01)

x = np.vstack((fix,x))
t = np.linspace(0.0,T,Nt)

# make data
data = np.random.normal(0.0,S,(Nt,N))
sigma = S*np.ones((Nt,N))

smooth,perts = pygeons.smooth.network_smoother(
                   data,t,x,sigma=sigma,
                   diff_specs=[pygeons.diff.VELOCITY_LAPLACIAN],
                   penalties=[P],perts=PERTS,procs=6)
smooth = pygeons.diff.diff(smooth,t,x,pygeons.diff.VELOCITY,procs=6)
perts = pygeons.diff.diff(perts,t,x,pygeons.diff.VELOCITY,procs=6)

# remove x axis
smooth = smooth[Nt//2,:]
perts = perts[:,Nt//2,:]

# compute correlation matrix
C = np.corrcoef(perts.T)
# extract correlation for just x=[0.0,0.0]
corr = C[0,:]

# plot the data
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=data[0,:]/S,cmap=myplot.cm.viridis,
      vmin=-2.0,vmax=2.0)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement (S)')

# plot smoothed data
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=smooth/S,cmap=myplot.cm.viridis,zorder=1)
#               vmin=-0.7,vmax=0.7)
ax.tripcolor(x[:,0]/L,x[:,1]/L,smooth/S,cmap=myplot.cm.viridis,zorder=0)
#               vmin=-0.7,vmax=0.7)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement (S)')

# plot correlation
fig,ax = plt.subplots()
c = ax.scatter(x[:,0]/L,x[:,1]/L,s=100,c=corr,cmap='seismic',zorder=1,
               vmin=-1.0,vmax=1.0)
ax.tripcolor(x[:,0]/L,x[:,1]/L,corr,cmap='seismic',zorder=0,
               vmin=-1.0,vmax=1.0)
ax.set_xlim((-5.5,5.5))
ax.set_ylim((-5.5,5.5))
ax.set_xlabel('position (L)')
ax.grid()
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('correlation')
plt.show()
