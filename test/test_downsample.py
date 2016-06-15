#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pygeons.smooth import network_smoother
from pygeons.diff import ACCELERATION
from scipy.spatial import cKDTree

class MeanInterpolant:
  '''   
  An interplant whose value at x is the mean of all values observed 
  within some radius of x
  '''
  def __init__(self,x,value,sigma=None):
    ''' 
    Parameters
    ----------
      x : (N,D) array

      value : (N,...) array

      sigma : (N,...) array

    '''
    x = np.asarray(x)
    value = np.asarray(value)
    if sigma is None:
      sigma = np.ones(value.shape)

    # form observation KDTree 
    self.Tobs = cKDTree(x)
    self.value = value
    self.sigma = sigma
    self.value_shape = value.shape[1:]

  def __call__(self,xitp,radius):
    ''' 
    Parameters
    ----------
      x : (K,D) array

      radius : scalar

    Returns
    -------  
      out_value : (K,...) array

      out_sigma : (K,...) array
    '''
    xitp = np.asarray(xitp)
    Titp = cKDTree(xitp)
    idx_arr = Titp.query_ball_tree(self.Tobs,radius)
    out_value = np.zeros((xitp.shape[0],)+self.value_shape)
    out_sigma = np.zeros((xitp.shape[0],)+self.value_shape)
    for i,idx in enumerate(idx_arr):
      numer = np.sum(self.value[idx]/self.sigma[idx]**2,axis=0)
      denom = np.sum(1.0/self.sigma[idx]**2,axis=0)
      out_value[i] = numer/denom
      out_sigma[i] = np.sqrt(1.0/denom)

    return out_value,out_sigma


def rms(x):
  ''' 
  root mean squares
  '''
  x = np.asarray(x)
  return np.sqrt(np.sum(x**2)/x.shape[0])
  
Nsmall = 500
F = 10
Nbig = Nsmall*F
S = 1.0
T = 0.3

P = [(T/2.0)**2/S]


tsmall = np.linspace(0.0,1.0,Nsmall)
tbig = np.linspace(0.0,1.0,Nbig)

usmall = np.random.normal(0.0,S,(Nsmall,1))
ssmall = np.ones((Nsmall,1))

ubig = np.zeros((Nsmall,F))
sbig = np.zeros((Nsmall,F)) + np.inf

ubig[:,0] = usmall[:,0]
sbig[:,0] = ssmall[:,0]

ubig = ubig.ravel()
sbig = sbig.ravel()
ubig = ubig[:,None]
sbig = sbig[:,None]

x = np.array([[0.0,0.0]])

P = [(T/2.0)**2*rms(1.0/ssmall)]
print(P)
smooth_small,perts = network_smoother(
                       usmall,tsmall,x,sigma=ssmall,
                       diff_specs=[ACCELERATION],
                       penalties=P)

P = [(T/2.0)**2*rms(1.0/sbig)]
#P = [(1.0/np.sqrt(F))*P[0]]
print(P)
#P = [(T/2.0)**2*np.linalg.norm(1.0/sbig)]
#print(P)
smooth_big,perts = network_smoother(
                       ubig,tbig,x,sigma=sbig,
                       diff_specs=[ACCELERATION],
                       penalties=P)

fig,ax = plt.subplots()
ax.plot(tsmall,smooth_small,'k-')   
ax.plot(tbig,smooth_big,'r-')   
#plt.plot(tbig,ubig,'r.')
plt.show()

quit()
N = 1000
dt = 0.1/10
T = 0.1
S = 1.0

t = np.linspace(0.0,1.0,N)
x = np.array([[0.0,0.0]])
u = np.random.normal(0.0,S,(N,1))
sigma = S*np.ones((N,1))
I = MeanInterpolant(t[:,None],u,sigma=sigma)

tds = np.arange(0.0,1.0,dt)
uds,sds = I(tds[:,None],dt/2.0)
print(uds.shape)
print(sds.shape)

P = [(T/2.0)**2/np.mean(sds)]
udss,perts = network_smoother(uds,tds,x,sigma=sds,
                              diff_specs=[ACCELERATION],
                              penalties=P)

fig,ax = plt.subplots()
#ax.plot(t,u,'k.')
ax.plot(t,us,'k-')
#ax.plot(tds,uds,'b.')
ax.plot(tds,udss,'b-')
plt.show()
