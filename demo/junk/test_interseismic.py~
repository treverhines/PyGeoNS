#!/usr/bin/env python
import numpy as np
import rbf.halton
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import myplot.cm
import pygeons.smooth
import pygeons.diff
import logging
from myplot.colorbar import pseudo_transparent_cmap
logging.basicConfig(level=logging.INFO)

def animate_scalar(u,t,x,title=''):
  fig,ax = plt.subplots()
  ax.set_title(title)
  vmax = np.max(u)
  vmin = np.min(u)
  aviridis = pseudo_transparent_cmap(myplot.cm.viridis,0.5)
  ax.tripcolor(x[:,0],x[:,1],0.0*u[0,:],
               vmin=vmin,vmax=vmax,
               cmap=aviridis)
  c = ax.scatter(x[:,0],x[:,1],
                 s=100,c=0.0*u[0,:],vmin=vmin,vmax=vmax,
                 edgecolor='k',cmap=myplot.cm.viridis)
  cbar = fig.colorbar(c)
  cbar.set_label('displacement')

  def animate(i):
    ax.clear()
    ax.set_title(title)
    ax.tripcolor(x[:,0],x[:,1],u[i,:],
                 vmin=vmin,vmax=vmax,
                 cmap=aviridis)
    ax.scatter(x[:,0],x[:,1],s=100,c=u[i,:],vmin=vmin,vmax=vmax,
    edgecolor='k',cmap=myplot.cm.viridis)
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

def animate_quiver(u1,u2,t,x,title=''):
  fig,ax = plt.subplots()
  ax.set_title(title)
  q = ax.quiver(x[:,0],x[:,1],u1[0,:],u2[0,:],scale=20.0)
  ax.quiverkey(q,0.8,0.05,1.0,'1 unit')
  def animate(i):
    ax.clear()
    ax.set_title(title)
    ax.quiver(x[:,0],x[:,1],u1[i,:],u2[i,:],scale=20.0)
    ax.quiverkey(q,0.8,0.05,1.0,'1 unit')
    ax.text(np.min(x[:,0])-0.2*np.std(x[:,0]),
            np.min(x[:,1])-0.2*np.std(x[:,1]),
            'time: ' + str(np.round(t[i],2)))
    return ()

  def init():
    ax.clear()
    return ()

  ani = animation.FuncAnimation(fig, animate,len(t),init_func=init,
                                interval=100, blit=True)

  return ani


T = 1.0
L = 2.5
S = 0.1
Nt = 10
Nx = 100

t = np.linspace(0.0,1.0,Nt)
x = 100*(rbf.halton.halton(Nx,2) - 0.5)

u1 = 0.0*x[:,1][None,:] * t[:,None]
u2 = np.arctan(x[:,0]/10.0)[None,:] * t[:,None]

u1 += np.random.normal(0.0,S,u1.shape)
u2 += np.random.normal(0.0,S,u2.shape)

ds = [pygeons.diff.ACCELERATION,
      pygeons.diff.VELOCITY_LAPLACIAN]

penalties = [(T/2.0)**2/S,(T/2.0)*(L/2.0)**2/S]

u1_smooth,u1_pert = pygeons.smooth.network_smoother(u1,t,x,diff_specs=ds,
                                                    penalties=penalties,perts=0,
                                                    solve_ksp='preonly',solve_pc='lu')
u2_smooth,u2_pert = pygeons.smooth.network_smoother(u2,t,x,diff_specs=ds,
                                                    penalties=penalties,perts=0,
                                                    solve_ksp='preonly',solve_pc='lu')

v1dx = pygeons.diff.diff(u1_smooth,t,x,pygeons.diff.VELOCITY_DX)
v2dx = pygeons.diff.diff(u2_smooth,t,x,pygeons.diff.VELOCITY_DX)
v1dy = pygeons.diff.diff(u1_smooth,t,x,pygeons.diff.VELOCITY_DY)
v2dy = pygeons.diff.diff(u2_smooth,t,x,pygeons.diff.VELOCITY_DY)

e11 = v1dx
e22 = v2dy
e12 = 0.5*(v1dy + v2dx)

I2 = -(e11*e22 - e12**2)
#fig,ax = plt.subplots()
#ax.quiver(x[:,0],x[:,1],u1,u2)
ani1 = animate_quiver(u1,u2,t,x)
ani2 = animate_quiver(u1_smooth,u2_smooth,t,x)
ani3 = animate_scalar(I2,t,x)
plt.show()
