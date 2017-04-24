import numpy as np
import logging
from pygeons.main.gptools import composite
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from pygeons.main.strain import _station_sigma_and_p
from rbf.gauss import _cholesky_block_inv
logger = logging.getLogger(__name__)

def _fit(d,s,mu,sigma,p):
  ''' 
  conditions the discrete Gaussian process described by *mu*, *sigma*,
  and *p* with the observations *d* which have uncertainty *s*.
  Returns the mean and standard deviation of the posterior.
  '''  
  n,m = p.shape
  Kinv = _cholesky_block_inv(sigma+np.diag(s**2),p)
  r = np.empty(n+m)
  r[:n] = d - mu
  r[n:] = 0.0
  k = np.empty((n,n+m))
  k[:,:n] = sigma
  k[:,n:] = p
  u = mu + k.dot(Kinv.dot(r))
  su = np.sqrt(np.diag(sigma) - np.sum(k*Kinv.dot(k.T).T,axis=1))
  return u,su


def fit(t,x,d,sd,
        network_model=('se-se',),
        network_params=(1.0,0.05,50.0),
        station_model=('p0',),
        station_params=()):
  ''' 
  Returns the condition Gaussian process evaluated at the data points

  Parameters
  ----------
  t : (Nt,) array
  x : (Nx,2) array
  d : (Nt,Nx) array
  s : (Nt,Nx) array
  network_model : str array
  network_params : float array
  station_model : str array
  station_params : float array
  '''
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float,copy=True)
  sd = np.array(sd,dtype=float,copy=True)

  net_gp = composite(network_model,network_params,gpnetwork.CONSTRUCTORS)
  sta_gp = composite(station_model,station_params,gpstation.CONSTRUCTORS)

  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')
  # flat observation times and positions
  z = np.array([t_grid.ravel(),
                x0_grid.ravel(),
                x1_grid.ravel()]).T

  # mask indicates missing data
  mask = np.isinf(sd)
  z,d,sd = z[~mask.ravel()],d[~mask],sd[~mask]

  # Build covariance and basis vectors for the combined process. Do
  # not evaluated at masked points
  full_sigma,full_p = _station_sigma_and_p(sta_gp,t,mask)
  full_sigma += net_gp.covariance(z,z)
  full_p = np.hstack((full_p,net_gp.basis(z)))
  # all processes are assumed to have zero mean
  full_mu = np.zeros(z.shape[0])

  # returns the indices of outliers 
  uf,suf = _fit(d,sd,full_mu,full_sigma,full_p)

  # best fit combination of signal and noise to the observations
  u = np.full((t.shape[0],x.shape[0]),np.nan)
  u[~mask] = uf
  su = np.full((t.shape[0],x.shape[0]),np.inf)
  su[~mask] = suf
  return u,su
