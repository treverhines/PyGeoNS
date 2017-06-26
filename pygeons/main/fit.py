''' 
Contains a function for fitting a Gaussian process to the
observations.
'''
import numpy as np
import scipy.sparse as sp
import logging
from pygeons.main.gptools import (composite,
                                  station_sigma_and_p)
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from rbf.gauss import (_as_sparse_or_array,
                       _as_covariance,
                       _PartitionedPosDefSolver)
logger = logging.getLogger(__name__)


def _fit(d,s,mu,sigma,p):
  ''' 
  conditions the discrete Gaussian process described by *mu*, *sigma*,
  and *p* with the observations *d* which have uncertainty *s*.
  Returns the mean and standard deviation of the posterior at the
  observation points.  
  '''  
  n,m = p.shape
  # *A* is the Gaussian process covariance with the noise
  # covariance added
  A = _as_sparse_or_array(sigma + _as_covariance(s))
  Ksolver = _PartitionedPosDefSolver(A,p)
  # compute mean of the posterior 
  vec1,vec2 = Ksolver.solve(d - mu,np.zeros(m)) 
  u = mu + sigma.dot(vec1) + p.dot(vec2)   
  # compute std. dev. of the posterior
  if sp.issparse(sigma):
    sigma = sigma.A

  mat1,mat2 = Ksolver.solve(sigma.T,p.T)
  del A,Ksolver
  #  # just compute the diagonal components of the covariance matrix
  #  # note that A.dot(B).diagonal() == np.sum(A*B.T,axis=1)
  su = np.sqrt(sigma.diagonal() - 
               np.sum(sigma*mat1.T,axis=1) -
               np.sum(p*mat2.T,axis=1))
                 
  return u,su


def fit(t,x,d,sd,
        network_model,
        network_params,
        station_model,
        station_params):
  ''' 
  Fit network and station processes to the observations, not
  distinguishing between signal and noise.
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
  sta_sigma,sta_p = station_sigma_and_p(sta_gp,t,mask)
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
