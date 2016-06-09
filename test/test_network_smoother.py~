#!/usr/bin/env python
import numpy as np
import rbf.halton
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pygeons.diff import ACCELERATION, DISPLACEMENT_LAPLACIAN, VELOCITY_LAPLACIAN
from pygeons.smooth import network_smoother
import time
import logging
logger = logging.basicConfig(level=logging.INFO)
np.random.seed(1)

def animate(u,t,x,title=''):
  vmin = -1.2
  vmax = 1.2
  fig,ax = plt.subplots()
  ax.set_title(title)
  c = ax.scatter(x[:,0],x[:,1],
                 s=200,c=0.0*u[0,:],
                 edgecolor='k',
                 vmin=vmin,vmax=vmax)
  cbar = fig.colorbar(c)
  cbar.set_label('displacement')
  
  def animate(i):
    ax.clear()
    ax.set_title(title)
    ax.scatter(x[:,0],x[:,1],s=200,c=u[i,:],
    edgecolor='k',
    vmin=vmin,vmax=vmax)
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

## test network smoother for Nx=1
#####################################################################
Nt = 100
Nx = 1

t = np.linspace(0.0,1.0,Nt)
x = np.random.random((Nx,2))
u_true = np.sin(2*np.pi*t)[:,None]
u_true = u_true.repeat(Nx,axis=1)
u = u_true + np.random.normal(0.0,0.1,(Nt,Nx))
sigma = 0.1*np.ones((Nt,Nx))

start_time = time.time()
u_smooth,u_pert = network_smoother(
                    u,t,x,
                    sigma=sigma,
                    diff_specs=[ACCELERATION],
                    cv_plot=True)
end_time = time.time()       

print('total run time for network_smoother: %s milliseconds' % 
      np.round((end_time - start_time)*1000,3))
      
u_std = np.std(u_pert,axis=0)

fig,ax = plt.subplots()
ax.plot(t,u,'k.')
ax.plot(t,u_smooth[:,0],'b-')
ax.plot(t,u_true[:,0],'r--')
ax.set_xlabel('time')
ax.set_ylabel('displacement')
ax.grid()
ax.fill_between(t,u_smooth[:,0]-u_std[:,0],u_smooth[:,0]+u_std[:,0],color='b',alpha=0.2)
ax.legend(['observed','smoothed','true'],frameon=False)
fig.tight_layout()

## test network smoother for Nt=1
#####################################################################
Nt = 1
Nx = 100
t = np.zeros(Nt)
x = rbf.halton.halton(Nx,2)

u_true = np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])[None,:]
u_true = u_true.repeat(Nt,axis=0)
u = u_true + np.random.normal(0.0,0.3,(Nt,Nx))
sigma = 0.3*np.ones((Nt,Nx))

start_time = time.time()
u_smooth,u_pert = network_smoother(
                    u,t,x,
                    sigma=sigma,
                    diff_specs=[DISPLACEMENT_LAPLACIAN],
                    cv_plot=True)
end_time = time.time()       

print('total run time for network_smoother: %s milliseconds' % 
      np.round((end_time - start_time)*1000,3))

u_std = np.std(u_pert,axis=0)

fig,ax = plt.subplots()
c =ax.scatter(x[:,0],x[:,1],s=200,c=u_smooth[0,:],vmin=-1.2,vmax=1.2)
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement')
ax.set_title('smoothed')
fig.tight_layout()

fig,ax = plt.subplots()
c = ax.scatter(x[:,0],x[:,1],s=200,c=u[0,:],vmin=-1.2,vmax=1.2)
cbar = plt.colorbar(c,ax=ax)
cbar.set_label('displacement')
ax.set_title('observed')
fig.tight_layout()

# test network smoother for Nt=100 and Nx=50
#####################################################################
Nt = 50
Nx = 50
t = np.linspace(0.0,1.0,Nt)
x = rbf.halton.halton(Nx,2)

u_true = np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])[None,:]*np.sin(2*np.pi*t)[:,None]
u = u_true + np.random.normal(0.0,0.3,(Nt,Nx))
sigma = 0.3*np.ones((Nt,Nx))

start_time = time.time()
u_smooth,u_pert = network_smoother(
                    u,t,x,
                    sigma=sigma,
                    diff_specs=[ACCELERATION,DISPLACEMENT_LAPLACIAN],
                    cv_bounds=[[-4.0,0.0],[-4.0,0.0]],
                    cv_plot=True)
end_time = time.time()       

anim1 = animate(u,t,x,'observed')
anim2 = animate(u_smooth,t,x,'smoothed')
plt.show()

