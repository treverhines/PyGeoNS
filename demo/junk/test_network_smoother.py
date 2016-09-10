#!/usr/bin/env python
import numpy as np
import rbf.halton
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import myplot.cm
import pygeons.diff
from pygeons.smooth import smooth
import time
import logging
logger = logging.basicConfig(level=logging.INFO)
np.random.seed(2)

def animate(u,t,x,title=''):
  fig,ax = plt.subplots()
  ax.set_title(title)
  vmax = np.max(u)
  vmin = np.min(u)
  c = ax.scatter(x[:,0],x[:,1],
                 s=200,c=0.0*u[0,:],vmin=vmin,vmax=vmax,
                 edgecolor='k',cmap=myplot.cm.viridis)
  #vmin = c.get_clim()[0]                 
  #vmax = c.get_clim()[1]                 
  cbar = fig.colorbar(c)
  cbar.set_label('displacement')
  
  def animate(i):
    ax.clear()
    ax.set_title(title)
    ax.scatter(x[:,0],x[:,1],s=200,c=u[i,:],vmin=vmin,vmax=vmax,
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

## test network smoother for Nx=1
#####################################################################
# time scale 
T = 0.1
S = 0.1
Nt = 1000
Nx = 1

t = np.linspace(0.0,1.0,Nt)
x = np.random.random((Nx,2))
u_true = np.sin(2*np.pi*t)[:,None]
u_true = u_true.repeat(Nx,axis=1)
u = u_true + np.random.normal(0.0,S,(Nt,Nx))
sigma = S*np.ones((Nt,Nx))

start_time = time.time()
u_smooth = smooth(t,x,u,
                  time_scale=0.1,
                  sigma=sigma)
end_time = time.time()       

print('total run time for network_smoother: %s milliseconds' % 
      np.round((end_time - start_time)*1000,3))
      
#u_std = np.std(u_pert,axis=0)


fig,ax = plt.subplots()
ax.plot(t,u,'k.')
ax.plot(t,u_smooth[:,0],'b-')
ax.plot(t,u_true[:,0],'r--')
ax.set_xlabel('time')
ax.set_ylabel('displacement')
ax.grid()
#ax.fill_between(t,u_smooth[:,0]-u_std[:,0],u_smooth[:,0]+u_std[:,0],color='b',alpha=0.2)
#ax.legend(['observed','smoothed','true'],frameon=False)
fig.tight_layout()
plt.show()
quit()
## test network smoother for Nt=1
#####################################################################
L = 0.1
S = 0.5
Nt = 1
Nx = 1000
t = np.zeros(Nt)
x = rbf.halton.halton(Nx,2)

u_true = np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])[None,:]
u_true = u_true.repeat(Nt,axis=0)
sigma = S*np.ones((Nt,Nx))
u = u_true + np.random.normal(0.0,sigma)

start_time = time.time()
u_smooth = network_smoother(
                    u,t,x,
                    length_scale=L,
                    sigma=sigma)
end_time = time.time()       

print('total run time for network_smoother: %s milliseconds' % 
      np.round((end_time - start_time)*1000,3))

#u_std = np.std(u_pert,axis=0)

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
plt.show()

# test network smoother for Nt=100 and Nx=50
#####################################################################
T = 0.5
L = 0.5
S = 1.0
Nt = 100
Nx = 200

t = 1.0*np.linspace(0.0,2.0*np.pi,Nt)
x = 2*np.pi*rbf.halton.halton(Nx,2)

u_true = (np.sin(x[:,0])*np.sin(x[:,1])[None,:]*
          np.sin(t)[:,None])
sigma = S*np.ones((Nt,Nx))
u = u_true + np.random.normal(0.0,sigma)

start_time = time.time()
u_smooth = network_smoother(
                    u,t,x,
                    length_scale=L,
                    time_scale=T,
                    sigma=sigma,
                    diff_specs=[pygeons.diff.acc(),
                                pygeons.diff.disp_laplacian()])
end_time = time.time()       

print('total run time for network_smoother: %s milliseconds' % 
      np.round((end_time - start_time)*1000,3))

#u_true = pygeons.diff.diff(u_true,t,x,pygeons.diff.vel())
#u_smooth = pygeons.diff.diff(u_smooth,t,x,pygeons.diff.vel())
#u_pert = pygeons.diff.diff(u_pert,t,x,pygeons.diff.vel())

end_time = time.time()       
anim1 = animate(u,t,x,'observed displacement')
anim2 = animate(u_smooth,t,x,'smoothed velocity')
fig,ax = plt.subplots()
plt.plot(t,u_smooth[:,10],'b-')
plt.plot(t,u_true[:,10],'k-')
plt.plot(t,u[:,10],'k.')
plt.show()
