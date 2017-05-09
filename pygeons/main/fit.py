''' 
Contains a function for fitting a Gaussian process to the
observations.
'''
import numpy as np
import logging
from pygeons.main.gptools import composite
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from pygeons.main.strain import _station_sigma_and_p
from rbf.gauss import (_as_sparse_or_array,
                       _as_covariance,
                       _InversePartitioned)
logger = logging.getLogger(__name__)


def _fit(d,s,mu,sigma,p):
  ''' 
  conditions the discrete Gaussian process described by *mu*, *sigma*,
  and *p* with the observations *d* which have uncertainty *s*.
  Returns the mean of the posterior.
  '''  
  n,m = p.shape
  # *sigma_and_s* is the Gaussian process covariance with the noise
  # covariance added
  sigma_and_s = _as_sparse_or_array(sigma + _as_covariance(s))
  Kinv = _InversePartitioned(sigma_and_s,p)
  # compute mean of the posterior 
  vec1,vec2 = Kinv.dot(d - mu,np.zeros(m)) 
  u = mu + sigma.dot(vec1) + p.dot(vec2)   
  # dont bother computing the uncertainties and just return zero for
  # now
  su = np.zeros_like(u)
  # compute std. dev. of the posterior
  #  mat1,mat2 = Kinv.dot(sigma.T,p.T)
  #  del Kinv
  #  # just compute the diagonal components of the covariance matrix
  #  # note that A.dot(B).diagonal() == np.sum(A*B.T,axis=1)
  #  su = np.sqrt(sigma.diagonal() - 
  #               np.sum(sigma*mat1.T,axis=1) -
  #               np.sum(p*mat2.T,axis=1))
  return u,su


def fit(t,x,d,sd,
        network_model=('se-se',),
        network_params=(5.0,0.05,50.0),
        station_model=('p0','p1'),
        station_params=()):
  ''' 
  Returns the mean of the conditioned Gaussian process evaluated at
  the data points. This is a quick calculation used to assess whether
  the Gaussian process is actually able to describe the observations.

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
  d = np.array(d,dtype=float)
  sd = np.array(sd,dtype=float)
  diff = np.array([0,0,0])

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
  sta_sigma,sta_p = _station_sigma_and_p(sta_gp,t,mask)
  net_sigma = net_gp._covariance(z,z,diff,diff)
  net_p = net_gp._basis(z,diff)
  # combine station gp with the network gp
  mu = np.zeros(z.shape[0])
  sigma = _as_sparse_or_array(sta_sigma + net_sigma)
  p = np.hstack((sta_p,net_p))
  del sta_sigma,net_sigma,sta_p,net_p
  # best fit combination of signal and noise to the observations
  uf,suf = _fit(d,sd,mu,sigma,p)
  # fold back into 2d arrays
  u = np.full((t.shape[0],x.shape[0]),np.nan)
  u[~mask] = uf
  su = np.full((t.shape[0],x.shape[0]),np.inf)
  su[~mask] = suf
  return u,su
