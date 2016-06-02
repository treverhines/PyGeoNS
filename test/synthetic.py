#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dislocation.source import rectangular
from myplot.cm import viridis
import rbf.halton
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
  q = ax.quiver(x[:,0],x[:,1],u[0,:],v[0,:],scale=2.0)
  ax.quiverkey(q,0.1,0.1,0.1,'0.1m')
  def animate(i):
    ax.clear()
    q = ax.quiver(x[:,0],x[:,1],u[i,:],v[i,:],scale=2.0)
    ax.quiverkey(q,0.1,0.1,0.1,'0.1m')
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
Ns = 1000
# number of time steps
Nt = 500

# fault geometry
strike = 0.0
dip = 90.0
top_center = [0.0,0.0,0.0]
width  = 1.0
length = 1.0
slip = [10.0,0.0,0.0]



# create synthetic data
#####################################################################
#pnts = 3*(rbf.halton.halton(Ns,2) - 0.51251241124) 
pnts = np.random.normal(0.0,5.0,(Ns,2))
# add z component, which is zero
pnts = np.hstack((pnts,np.zeros((Ns,1))))

# coseismic displacements
cdisp,cderr = rectangular(pnts,slip,top_center,length,width,strike,dip)
# interseismic velocities
idisp,cderr = rectangular(pnts,slip,[0.0,0.0,-5.0],1000.0,1000.0,strike,90.0)

t = np.linspace(0.0,5.0,Nt) 

# interseismic displacement 
disp = 0.01*t[:,None,None]**2*idisp
# dislocation at t=0.5
disp[t>2.5,:,:] += 0.0*cdisp

# add noise
#disp += np.random.normal(0.0,0.01,disp.shape)

# add correlated noise to each station
disp_true = np.copy(disp)
print('adding noise')
for n in range(Ns):
  for j in range(3):
    disp[:,n,j] += correlated_noise(0.05**2,0.5,t)

print('finished')

# mask some of the data
sigma = 0.05*np.ones((Nt,Ns))
#sigma[t<1.0,10] = np.inf
#disp[t<1.0,10,:] = 0.0


# smooth displacement
pred1,sigma_pred1 = pygeons.smooth.network_smoother(
                      disp[:,:,0],t,pnts[:,[0,1]],
                      sigma=sigma,procs=Nprocs,stencil_size=5,
#                      cv_plot=True,cv_itr=200,
                      cv_space_bounds=[0.0,4.0],cv_time_bounds=[0.0,4.0],
                      reg_space_parameter=10.0,
                      reg_time_parameter=1.0,
                      bs_itr=0,solve_view=False,
                      solve_max_itr=10000,solve_atol=1e-6,solve_rtol=1e-10,solve_ksp='lgmres',solve_pc='icc')
#                      stencil_time_smp=[[0]],stencil_time_vert=[[2.5]],
#                      stencil_space_smp=[[0,1]],stencil_space_vert=[[0.0,-0.5],[0.0,0.5]])
pred2,sigma_pred2 = pygeons.smooth.network_smoother(
                      disp[:,:,1],t,pnts[:,[0,1]],
                      sigma=sigma,procs=Nprocs,
#                      cv_plot=True,cv_itr=200,stencil_size=5,
                      reg_space_parameter=10.0,
                      reg_time_parameter=1.0,
                      bs_itr=0,solve_view=False,
                      solve_max_itr=10000,solve_atol=1e-6,solve_rtol=1e-10,solve_ksp='lgmres',solve_pc='icc')
#                      cv_space_bounds=[0.0,4.0],cv_time_bounds=[0.0,4.0],
#                      solve_atol=1e-4)
#                      stencil_time_smp=[[0]],stencil_time_vert=[[2.5]],
#                      stencil_space_smp=[[0,1]],stencil_space_vert=[[0.0,-0.5],[0.0,0.5]])

#fig,ax = plt.subplots()
#ax.quiver(pnts[:,0],pnts[:,1],cdisp[:,0],cdisp[:,1],color='r',scale=10.0)
#ax.quiver(pnts[:,0],pnts[:,1],idisp[:,0],idisp[:,1],color='k',scale=10.0)
#a1 = animate_it(disp[:,:,0],t,pnts)
a1 = animate_quiver(disp[:,:,0],disp[:,:,1],t,pnts)
a2 = animate_quiver(disp_true[:,:,0],disp_true[:,:,1],t,pnts)
a3 = animate_quiver(pred1,pred2,t,pnts)
print(pred1[25,10])
print(pred2[25,10])
print(pred1[35,19])
print(pred2[35,19])
#a3 = animate_it(disp[:,:,2],t,pnts)
#idx = 10
#print(pnts[idx])
#fig,ax = plt.subplots()
#ax.plot(t,disp[:,idx,1],'k.')
#ax.plot(t,disp_true[:,idx,1],'k--')
#ax.fill_between(t,pred2[:,idx]+sigma_pred2[:,idx],pred2[:,idx]-sigma_pred2[:,idx],color='b',alpha=0.3,edgecolor='none')
#ax.plot(t,pred2[:,idx],'b-')

plt.show()





