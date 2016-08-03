#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree
import logging
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
    
    data : (...,N) array
  '''
  N = data.shape[-1]
  bcast_shape = data.shape[:-1] 
  M = np.prod(bcast_shape)
  data = data.reshape((M,N))
  data = data.T  

  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)

  # do not use umfpack because it raises an error when the memory 
  # usage exceeds ~4GB. It is also not easy to catch when umfpack 
  # fails due to a singular matrix
  soln = scipy.sparse.linalg.spsolve(lhs,rhs,use_umfpack=False)
  if np.any(np.isnan(soln)):
    # spsolve fills the solution vector with nans when the matrix is 
    # singular
    raise ValueError(
''' 
Singular matrix. Possible causes include but are not limited to: 
         
  - having zeros in the uncertainty array

  - having multiple stations with the same position

  - not having enough data for there to be a unique solution
    at an interpolating/extrapolation point. This only causes a 
    singular matrix when *fill* is "interpolate" or "extrapolate"
             
  - having zero for *time_scale* or *length_scale*. This only
    causes a singular matrix when *fill* is "interpolate" or 
    "extrapolate"
  
''')     

  # reshape the solution to the original dimension
  soln = soln.T
  soln = soln.reshape(bcast_shape+(N,))
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
  if diff_specs['space']['diffs'] is None:
    xord = 0
  else:  
    xords =  [sum(i) for i in diff_specs['space']['diffs']]
    if not all([i==xords[0] for i in xords]):
      raise ValueError('all space derivative terms must have the same order')

    xord = xords[0]
    
  # make sure all time derivative terms have the same order
  if diff_specs['time']['diffs'] is None:
    tord = 0
  else:
    tords =  [sum(i) for i in diff_specs['time']['diffs']]
    if not all([i==tords[0] for i in tords]):
      raise ValueError('all time derivative terms must have the same order')

    tord = tords[0]
    
  
  out = (T/(2*np.pi))**tord*(L/(2*np.pi))**xord*(1.0/S)
  return out  
  

def _collapse_sparse_matrix(A,idx):
  ''' 
  collapse A so that only rows idx and columns idx remain
  '''
  A.eliminate_zeros()
  A = A.tocoo()
  N = len(idx)
  full_index_to_collapsed_index = dict(zip(idx,range(N)))    
  rnew = [full_index_to_collapsed_index[i] for i in A.row]
  cnew = [full_index_to_collapsed_index[i] for i in A.col]
  out = scipy.sparse.csr_matrix((A.data,(rnew,cnew)),(len(idx),len(idx)))
  return out


def _fill_mask(t,x,sigma,kind):
  ''' 
  Returns an (Nt,Nx) boolean array identifying when and where a 
  smoothed estimate should be made. True indicates that the datum 
  should not be estimated. 
  
  kind :
    'none' : output at times and stations where data are not missing
    
    'interpolate' : interpolate at times with missing data

    'extrapolate' : output at all stations and times

  '''
  data_is_missing = np.isinf(sigma)
  
  if kind == 'none':
    mask = data_is_missing 

  elif kind == 'interpolate':
    mask = np.ones(sigma.shape,dtype=bool)
    for i,xi in enumerate(x):
      active_times = t[~data_is_missing[:,i]]
      if active_times.shape[0] > 0: 
        first_time = np.min(active_times)
        last_time = np.max(active_times)
        mask[:,i] = (t<first_time) | (t>last_time)
  
  elif kind == 'extrapolate':
    mask = np.zeros(sigma.shape,dtype=bool)
      
  else:
    raise ValueError('*kind* must be "none", "interpolate", or "extrapolate"')

  return mask
  
def smooth(t,x,u,
           sigma=None,
           diff_specs=None,
           length_scale=None,                      
           time_scale=None,
           fill='none'):
  ''' 
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    u : (...,Nt,Nx) array
    
    sigma : (Nt,Nx) array
    
    diff_specs : list of DiffSpecs isntances
    
    length_scale : float
    
    time_scale : float
    
    fill : str, {'none', 'interpolate', 'extrapolate'}
      Indicates when and where to make a smoothed estimate. 'none' : 
      output only where data is not missing. 'interpolate' : output 
      where data is not missing and where time interpolation is 
      possible. 'all' : output at all stations and times. Masked data 
      is returned as np.nan
      
  '''                                 
  u = np.array(u,dtype=float,copy=True)
  # convert any nans to zeros
  u[np.isnan(u)] = 0.0
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  Nx,Nt = x.shape[0],t.shape[0]
  bcast_shape = u.shape[:-2]

  if diff_specs is None:
    diff_specs = [pygeons.diff.acc(),
                  pygeons.diff.disp_laplacian()]
 
  if u.shape[-2:] != (Nt,Nx):
    raise TypeError('u must have shape (...,Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))
  
  else:
    sigma = np.asarray(sigma)
    
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  mask = _fill_mask(t,x,sigma,fill)

  # estimate length scale and time scale if not given
  default_time_scale,default_length_scale = _estimate_scales(t,x)
  if length_scale is None:
    length_scale = default_length_scale

  if time_scale is None:
    time_scale = default_time_scale

  # flatten only the last two axes
  u_flat = u.reshape(bcast_shape+(Nt*Nx,))
  sigma_flat = sigma.reshape(Nt*Nx)
  mask_flat = mask.reshape(Nt*Nx)

  # get rid of masked entries in u_flat and sigma_flat
  keep_idx, = np.nonzero(~mask_flat)
  u_flat = u_flat[...,keep_idx]
  sigma_flat = sigma_flat[keep_idx]

  # weigh u by the inverse of data uncertainty.
  u_flat = u_flat/sigma_flat
  
  # system matrix is the identity matrix scaled by data weight
  K = len(keep_idx)
  Gdata = 1.0/sigma_flat
  Grow,Gcol = range(K),range(K)
  G = scipy.sparse.csr_matrix((Gdata,(Grow,Gcol)),(K,K))

  # create a regularization matrix for each diff_spec
  Lsubs = [pygeons.diff.diff_matrix(t,x,d,mask=mask) for d in diff_specs]
  # create regularization penalty parameters
  penalties = [_penalty(time_scale,length_scale,sigma,d) for d in diff_specs]

  # Collapse the regularization matrices on the masked rows 
  logger.debug('collapsing regularization matrices ...')
  Lsubs = [_collapse_sparse_matrix(Li,keep_idx) for Li in Lsubs]
  logger.debug('done')

  # stack the regularization matrices
  L = scipy.sparse.vstack(p*r for p,r in zip(penalties,Lsubs))
  L.eliminate_zeros()
  
  logger.debug('solving for predicted displacements ...')
  u_pred = _solve(G,L,u_flat)
  logger.debug('done')

  # expand the solution to the original size
  out = np.zeros(bcast_shape+(Nt*Nx,))
  out[...,keep_idx] = u_pred
  out = out.reshape(bcast_shape+(Nt,Nx))
  out[...,mask] = np.nan    
  return out
  

