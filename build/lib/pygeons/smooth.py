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
             
  - having zero for *min_wavelength*. This only causes a singular 
    matrix when *fill* is "interpolate" or "extrapolate"
  
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


def _default_min_wavelength(x):
  ''' 
  returns a time and spatial scale which is 10x the average shortest 
  distance between observations. If the average distance cannot be 
  computed due to a lack of points then 1.0 is returned
  '''
  dx = _average_shortest_distance(x)
  if np.isinf(dx):
    return 1.0
  else:
    return 20*dx      

def _rms(x):
  ''' 
  root mean squares
  '''
  out = np.sqrt(np.sum(x**2)/x.size)
  # if the result is nan (due to zero sized x) then return 0.0
  out = np.nan_to_num(out)    
  return out

def _penalty(min_wavelength,sigma,diffs):
  D = np.shape(diffs)[-1]
  diffs = np.reshape(diffs,(-1,D))
  order = sum(diffs[0])
  sigma_rms = 1.0/_rms(1.0/sigma) # characteristic uncertainty 
  out = (min_wavelength/(2*np.pi))**order/sigma_rms
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


def _time_fill_mask(t,sigma,kind):
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
  Nt,Nx = sigma.shape
    
  if kind == 'none':
    mask = data_is_missing 

  elif kind == 'interpolate':
    mask = np.ones(sigma.shape,dtype=bool)
    for i in range(Nx):
      active_times = t[~data_is_missing[:,i],0]
      if active_times.shape[0] > 0: 
        first_time = np.min(active_times)
        last_time = np.max(active_times)
        mask[:,i] = (t[:,0]<first_time) | (t[:,0]>last_time)
  
  elif kind == 'extrapolate':
    mask = np.zeros(sigma.shape,dtype=bool)
      
  else:
    raise ValueError('*kind* must be "none", "interpolate", or "extrapolate"')

  return mask

def _space_fill_mask(x,sigma,kind):
  ''' 
  Returns an (Nt,Nx) boolean array identifying when and where a 
  smoothed estimate should be made. True indicates that the datum 
  should not be estimated. 
  
  kind :
    'none' : output at times and stations where data are not missing
    
    'extrapolate' : output at all stations and times

  '''
  data_is_missing = np.isinf(sigma)
  Nt,Nx = sigma.shape

  if kind == 'none':
    mask = data_is_missing 
  
  elif kind == 'extrapolate':
    mask = np.zeros(sigma.shape,dtype=bool)
      
  else:
    raise ValueError('*kind* must be "none" or "extrapolate"')

  return mask
  

def time_smooth(t,x,u,sigma=None,diffs=None,
                min_wavelength=None,
                fill='none',**kwargs):
  ''' 
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    u : (...,Nt,Nx) array
    
    diffs : (1,) or (K,1) array
      derivative order
    
    sigma : (Nt,Nx) array, optional
    
    min_wavelength : float, optional
      lowest wavelength allowed in the smoothed solution
    
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

  if u.shape[-2:] != (Nt,Nx):
    raise TypeError('u must have shape (...,Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))  
  else:
    sigma = np.asarray(sigma)
    
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  if min_wavelength is None:
    min_wavelength = _default_min_wavelength(t)

  if diffs is None:
    diffs = [[2]]
    
  # identify when/where smoothed data will not be estimated
  mask = _time_fill_mask(t,sigma,fill)

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

  # create a regularization matrix 
  L = pygeons.diff.time_diff_matrix(t,x,diffs,mask=mask,**kwargs)
  L = _collapse_sparse_matrix(L,keep_idx)
  # create regularization penalty parameters
  p = _penalty(min_wavelength,sigma,diffs)
  L *= p
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
  

def space_smooth(t,x,u,sigma=None,diffs=None,
                 min_wavelength=None,
                 fill='none',**kwargs):
  ''' 
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    u : (...,Nt,Nx) array
    
    diffs : (2,) or (K,2) array
      derivative order
    
    sigma : (Nt,Nx) array, optional
    
    min_wavelength : float, optional
      lowest wavelength allowed in the smoothed solution
    
    fill : str, {'none', 'extrapolate'}
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

  if u.shape[-2:] != (Nt,Nx):
    raise TypeError('u must have shape (...,Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))  
  else:
    sigma = np.asarray(sigma)
    
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  if min_wavelength is None:
    min_wavelength = _default_min_wavelength(x)

  if diffs is None:
    diffs = [[2,0],[0,2]]
    
  # identify when/where smoothed data will not be estimated
  mask = _space_fill_mask(x,sigma,fill)

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

  # create a regularization matrix 
  L = pygeons.diff.space_diff_matrix(t,x,diffs,mask=mask,**kwargs)
  L = _collapse_sparse_matrix(L,keep_idx)
  # create regularization penalty parameters
  p = _penalty(min_wavelength,sigma,diffs)
  L *= p
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
  

