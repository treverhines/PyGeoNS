#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dislocation.source import rectangular
from myplot.cm import viridis
import rbf.halton
import pygeons.cuts
import pygeons.smooth
import rbf.basis
import logging
logger = logging.basicConfig(level=logging.INFO)
np.random.seed(1)

def correlated_noise(var,decay,times):
  N = len(times)
  mean = np.zeros(N)
  t1,t2 = np.meshgrid(times,times)
  cov = var*np.exp(-np.abs(t1 - t2)/decay)
  noise = np.random.multivariate_normal(mean,cov,1)
  return noise[0]

def animate_scatter(u,t,x):
  fig,ax = plt.subplots()
  c = ax.scatter(x[:,0],x[:,1],
                s=200,c=0.0*u[0,:],
                cmap=viridis,edgecolor='k',
                vmin=np.min(u),vmax=np.max(u))
  fig.colorbar(c)

  def animate(i):
    ax.clear()
    ax.scatter(x[:,0],x[:,1],s=200,c=u[i,:],
    cmap=viridis,edgecolor='k',
    vmin=np.min(u),vmax=np.max(u))
    ax.text(np.min(x[:,0])-0.4*np.std(x[:,0]),
            np.min(x[:,1])-0.4*np.std(x[:,1]),
            'time: ' + str(np.round(t[i],2)))
    return ()

  def init():
    ax.clear()
    return ()

  ani = animation.FuncAnimation(fig, animate,len(t),init_func=init,
                                interval=100, blit=True)

  return ani

def animate_quiver(u,v,t,x):
  fig,ax = plt.subplots()
  q = ax.quiver(x[:,0],x[:,1],u[0,:],v[0,:],scale=1.0)
  ax.quiverkey(q,0.1,0.1,1.0,'1.0m')
  def animate(i):
    ax.clear()
    q = ax.quiver(x[:,0],x[:,1],u[i,:],v[i,:],scale=1.0)
    ax.quiverkey(q,0.1,0.1,1.0,'1.0m')
    #ax.scatter(x[:,0],x[:,1],s=200,c=u[i,:],
    #q.set_UVC(u[i,:],v[i,:])  
    #cmap=viridis,edgecolor='k',
    #vmin=np.min(u),vmax=np.max(u))
    ax.text(np.min(x[:,0])-0.4*np.std(x[:,0]),
            np.min(x[:,1])-0.4*np.std(x[:,1]),
            'time: ' + str(np.round(t[i],2)))
    return ()

  def init():
    ax.clear()
    return ()

  ani = animation.FuncAnimation(fig, animate,len(t),init_func=init,
                                interval=100, blit=True)

  return ani

Nprocs = 6

# number of stations
Ns = 20
# number of time steps
Nt = 100

# fault geometry
strike = 0.0
dip = 90.0
top_center = [0.0,0.0,0.0]
width  = 5.0
length = 40.0
slip = [10.0,0.0,0.0]



# create synthetic data
#####################################################################
#pnts = 50*(rbf.halton.halton(Ns,2) - 0.51251241124) 
pnts = np.random.normal(0.0,20.0,(Ns,2))
# add z component, which is zero
pnts = np.hstack((pnts,np.zeros((Ns,1))))

# coseismic displacements
cdisp,cderr = rectangular(pnts,slip,top_center,length,width,strike,dip)
# interseismic velocities
idisp,cderr = rectangular(pnts,slip,[0.0,0.0,-width],1000.0,1000.0,strike,90.0)

t = np.linspace(0.0,5.0,Nt) 

# interseismic displacement 
#disp = 0.1*np.sin(t[:,None,None])*idisp
disp = 0.1*np.sin(t[:,None,None])*idisp
# dislocation at t=0.5
disp[t>2.5,:,:] += 1.0*cdisp

# add noise
#disp += np.random.normal(0.0,0.01,disp.shape)

# add correlated noise to each station
disp_true = np.copy(disp)
print('adding noise')
for n in range(Ns):
  for j in range(3):
    disp[:,n,j] += correlated_noise(0.05**2,0.5,t)
    #disp[:,n,j] += np.random.normal(0.0,0.05,len(t))

print('finished')

# mask some of the data
sigma = 0.05*np.ones((Nt,Ns))
#sigma[t<1.0,10] = np.inf
#disp[t<1.0,10,:] = 0.0


time_cut = pygeons.cuts.TimeCut(2.5)
space_cut = pygeons.cuts.SpaceCut([0.0,-20],[0.0,20])
#space_cut = pygeons.cuts.SpaceCut()
#scc = pygeons.cuts.SpaceCutCollection([space_cut])
scc = pygeons.cuts.SpaceCutCollection()
tcc = pygeons.cuts.TimeCutCollection([time_cut])

# smooth displacement
pred1,pert1 = pygeons.smooth.network_smoother(
                      disp[:,:,0],t,pnts[:,[0,1]],
#                      reg_space_parameter=1e-10,
#                      reg_time_parameter=1.0,
                      stencil_time_cuts=tcc, 
                      stencil_space_cuts=scc, 
                      sigma=sigma,stencil_space_size=8,
                      cv_itr=200,cv_plot=True,cv_chunk='space',
                      baseline=False,procs=6,
#                      solve_ksp='preonly',solve_pc='lu',
                      solve_atol=1e-6,solve_max_itr=10000)
#pred1 = pygeons.diff.time_diff(pred1,t,pnts[:,[0,1]],diff=(1,),cuts=tcc)
#pred1 = pygeons.diff.space_diff(pred1,t,pnts[:,[0,1]],diff=(1,0))

pred2,pert2 = pygeons.smooth.network_smoother(
                      disp[:,:,1],t,pnts[:,[0,1]],
                      stencil_time_cuts=tcc, 
                      stencil_space_cuts=scc, 
#                      reg_space_parameter=1e-10,
#                      reg_time_parameter=1.0,
                      sigma=sigma,stencil_space_size=8,
                      cv_itr=200,cv_plot=True,cv_chunk='space',
                      procs=6,baseline=False,
#                      solve_ksp='preonly',solve_pc='lu',
                      solve_atol=1e-6,solve_max_itr=10000)
#pred2 = pygeons.diff.time_diff(pred2,t,pnts[:,[0,1]],diff=(1,),cuts=tcc)
#pred2 = pygeons.diff.space_diff(pred2,t,pnts[:,[0,1]],diff=(1,0))
#disp_true[:,:,0] = pygeons.diff.time_diff(disp_true[:,:,0],t,pnts[:,[0,1]],diff=(1,),cuts=tcc)
#disp_true[:,:,1] = pygeons.diff.time_diff(disp_true[:,:,1],t,pnts[:,[0,1]],diff=(1,),cuts=tcc)

a1 = animate_quiver(disp[:,:,0],disp[:,:,1],t,pnts)
a2 = animate_quiver(disp_true[:,:,0],disp_true[:,:,1],t,pnts)
a3 = animate_quiver(pred1,pred2,t,pnts)

fig,ax = plt.subplots()
ax.plot(t,disp[:,:,1],'b.-')
ax.plot(t,disp_true[:,:,1],'k.-')
ax.plot(t,pred2,'r.-')

plt.show()





