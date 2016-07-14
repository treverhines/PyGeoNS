#!/usr/bin/env python
from __future__ import division
import numpy as np
import modest.cv
import modest
import scipy.sparse
from scipy.spatial import cKDTree
import logging
import pygeons.cuts
import modest.mp
import pygeons.diff
logger = logging.getLogger(__name__)

def _solve(A,L,data):
  ''' 
  solves the sparse regularized least squares problem
  
    | A | m = | data |
    | L |     |  0   |
    
  and returns an error in the event of a singular gramian matrix

  Parameters
  ----------
    A : (N,M) csr sparse matrix
    
    L : (P,M) csr sparse matrix
    
    d : (N,) array
  '''  
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)
  # do not use umfpack because it raises an error when the memory 
  # usage exceeds ~4GB. It is also not easy to catch when umfpack 
  # fails due to a singular matrix
  soln = scipy.sparse.linalg.spsolve(lhs,rhs,use_umfpack=False)
  if np.any(np.isnan(soln)):
    # spsolve fills the solution vector with nans when the matrix is 
    # singular
    raise ValueError('Singular matrix. This may result from having too '
                     'many masked observations. Check for stations '
                     'or time periods where all observations are masked')
  return soln
  
def _average_shortest_distance(x):
  ''' 
  returns the average distance to nearest neighbor           

  Parameters
  ----------
    x : (N,D) array

  Returns
  -------
    out : float
  '''
  # if no points are given then the spacing is infinite
  if x.shape[0] == 0:
    return np.inf
    
  T = cKDTree(x)
  out = np.mean(T.query(x,2)[0][:,1])
  return out


def _estimate_scales(t,x):
  ''' 
  returns a time and spatial scale which is 10x the average shortest 
  distance between observations. If the average distance cannot be 
  computed due to a lack of points then 1.0 is returned
  '''
  dt = _average_shortest_distance(t[:,None])
  dl = _average_shortest_distance(x)
  T = 10*dt
  L = 10*dl
  if np.isinf(T):
    T = 1.0
  if np.isinf(L):
    L = 1.0
    
  return T,L


def _rms(x):
  ''' 
  root mean squares
  '''
  out = np.sqrt(np.sum(x**2)/x.size)
  # if the result is nan (due to zero sized x) then return 0.0
  out = np.nan_to_num(out)    
  return out


def _penalty(T,L,sigma,diff_specs):
  ''' 
  returns scaling parameter for the regularization constraint
  '''
  S = 1.0/_rms(1.0/sigma) # characteristic uncertainty 
  
  # make sure all space derivative terms have the same order
  xords =  [sum(i) for i in diff_specs['space']['diffs']]
  # make sure all time derivative terms have the same order
  tords =  [sum(i) for i in diff_specs['time']['diffs']]
  if not all([i==xords[0] for i in xords]):
    raise ValueError('all space derivative terms must have the same order')
  if not all([i==tords[0] for i in tords]):
    raise ValueError('all time derivative terms must have the same order')

  xord = xords[0]
  tord = tords[0]
  out = (T/2.0)**tord*(L/2.0)**xord/S
  return out  
  

def network_smoother(u,t,x,
                     sigma=None,
                     diff_specs=None,
                     length_scale=None,                      
                     time_scale=None,
                     procs=None,
                     perts=10):
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)

  Nx = x.shape[0]
  Nt = t.shape[0]

  if diff_specs is None:
    diff_specs = [pygeons.diff.acc(),
                  pygeons.diff.disp_laplacian()]

  if u.shape != (Nt,Nx):
    raise TypeError('u must have shape (Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))
  
  else:
    sigma = np.asarray(sigma)
    
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  u_flat = u.ravel()
  sigma_flat = sigma.ravel()

  reg_matrices = [pygeons.diff.diff_matrix(t,x,d,procs=procs) for d in diff_specs]

  # estimate length scale and time scale if not given
  default_time_scale,default_length_scale = _estimate_scales(t,x)
  if length_scale is None:
    length_scale = default_length_scale

  if time_scale is None:
    time_scale = default_time_scale
    
  # create regularization penalty parameters
  penalties = [_penalty(time_scale,length_scale,sigma,d) for d in diff_specs]

  # system matrix is the identity matrix scaled by data weight
  Gdata = 1.0/sigma_flat
  Grow = range(Nt*Nx)
  Gcol = range(Nt*Nx)
  Gsize = (Nt*Nx,Nt*Nx)
  G = scipy.sparse.csr_matrix((Gdata,(Grow,Gcol)),Gsize)
  
  # weigh u by the inverse of data uncertainty.
  u_flat = u_flat/sigma_flat

  # this makes matrix copies
  L = scipy.sparse.vstack(p*r for p,r in zip(penalties,reg_matrices))
  L.eliminate_zeros()
  
  logger.debug('solving for predicted displacements')
  u_pred = _solve(G,L,u_flat)
  logger.debug('done')

  logger.debug('computing perturbed predicted displacements')
  u_pert = np.zeros((perts,G.shape[0]))
  # perturbed displacements will be computed in parallel and so this 
  # needs to be turned into a mappable function
  def mappable_dsolve(args):
    G = args[0]
    L = args[1]
    d = args[2]
    return _solve(G,L,d)

  # generator for arguments that will be passed to calculate_pert
  args = ((G,L,np.random.normal(0.0,1.0,G.shape[0]))
           for i in range(perts))
  u_pert = modest.mp.parmap(mappable_dsolve,args,workers=procs)
  u_pert = np.reshape(u_pert,(perts,(Nt*Nx)))
  u_pert += u_pred[None,:]

  logger.debug('done')

  u_pred = u_pred.reshape((Nt,Nx))
  u_pert = u_pert.reshape((perts,Nt,Nx))

  return u_pred,u_pert


