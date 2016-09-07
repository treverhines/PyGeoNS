#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
import scipy.sparse
import logging
logger = logging.getLogger(__name__)

def time_diff_matrix(t,x,diffs,mask=None,cuts=None,**kwargs):
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]
  if mask is None:
    mask = np.zeros((Nt,Nx),dtype=bool)
  else:
    mask = np.asarray(mask,dtype=bool)

  if cuts is not None:
    vert = np.array(cuts).reshape((-1,1))
    smp = np.arange(vert.shape[0]).reshape((-1,1))
  else:
    vert = None
    smp = None  

  # A time differentiation matrix is created for each station in this 
  # loop. If two stations have the same mask and the same time cuts 
  # then reused the matrix. 
  cache = {}
  Lsubs = []
  for i,xi in enumerate(x):
    # create a tuple identifying the mask for this station 
    key = tuple(mask[:,i])
    if key in cache:
      Lsubs += [cache[key]]
      continue

    # find the indices of unmasked times for this station
    sub_idx, = np.nonzero(~mask[:,i])
    Li = rbf.fd.diff_matrix_1d(t[sub_idx],diffs,vert=vert,smp=smp,**kwargs)
      
    # convert to coo to get row and col indices for each entry
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    # changes the coordinates to correspond with the t vector rather 
    # than t[sub_idx]
    Li = scipy.sparse.coo_matrix((vi,(sub_idx[ri],sub_idx[ci])),shape=(Nt,Nt))

    cache[key] = Li             
    Lsubs += [Li]
    
  # combine submatrices into the master matrix
  count = 0
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
  nnz = sum(Li.nnz for Li in Lsubs)
  rows = np.zeros((nnz,),dtype=int)
  cols = np.zeros((nnz,),dtype=int)
  vals = np.zeros((nnz,),dtype=float)
  for i,Li in enumerate(Lsubs):
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[count:count+Li.nnz] = wrapped_indices[ri,i]
    cols[count:count+Li.nnz] = wrapped_indices[ci,i]
    vals[count:count+Li.nnz] = vi
    count += Li.nnz

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def space_diff_matrix(t,x,diffs,mask=None,cuts=None,**kwargs):
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]
  if mask is None:
    mask = np.zeros((Nt,Nx),dtype=bool)
  else:
    mask = np.asarray(mask,dtype=bool)

  if cuts is not None:
    vert = np.array(cuts).reshape((-1,2))
    smp = np.arange(vert.shape[0]).reshape((-1,2))
  else:
    vert = None
    smp = None  
    
  # diff matrices for a collection of space cuts are stored in 
  # this dictionary and then recalled if another matrix is to be 
  # generated with the the same space cuts
  cache = {}
  Lsubs = []
  for i,ti in enumerate(t):
    key = tuple(mask[i,:])
    if key in cache:
      Lsubs += [cache[key]]
      continue
      
    # find the indices of unmasked stations for this time
    sub_idx, = np.nonzero(~mask[i,:])
    Li = rbf.fd.diff_matrix(x[sub_idx],diffs,vert=vert,smp=smp,**kwargs)

    # convert to coo to get row and col indices for each entry
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    # changes the coordinates to correspond with the x vector rather 
    # than x[sub_idx]
    Li = scipy.sparse.coo_matrix((vi,(sub_idx[ri],sub_idx[ci])),shape=(Nx,Nx))

    cache[key] = Li
    Lsubs += [Li]
             
  # combine submatrices into the master matrix
  count = 0
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
  nnz = sum(Li.nnz for Li in Lsubs)
  rows = np.zeros((nnz,),dtype=int)
  cols = np.zeros((nnz,),dtype=int)
  vals = np.zeros((nnz,),dtype=float)
  for i,Li in enumerate(Lsubs):
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[count:count+Li.nnz] = wrapped_indices[i,ri]
    cols[count:count+Li.nnz] = wrapped_indices[i,ci]
    vals[count:count+Li.nnz] = vi
    count += Li.nnz

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def time_diff(t,x,u,diffs,mask=None,cuts=None,**kwargs):
  ''' 
  differentiates u
  
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    u : (...,Nt,Nx) array 

    diffs : 
    
    mask : (Nt,Nx) array
      Identifies which elements of u to ignore. This is incase there 
      are outliers or missing data. The returned diffentiated array 
      will have np.nan where the mask is True
      
    kwargs : dict
      passed to time_diff_matrix
            
  Returns
  -------
    u_diff : (Nt,Nx) array
    
  '''
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  u = np.asarray(u,dtype=float)
  Nt,Nx = t.shape[0],x.shape[0]
  if mask is None:
    mask = np.zeros((Nt,Nx),dtype=bool)
  else:
    mask = np.asarray(mask,dtype=bool)
        
  bcast_shape = u.shape[:-2]
  M = np.prod(bcast_shape)

  D = time_diff_matrix(t,x,diffs,mask=mask,cuts=cuts,**kwargs)
  u_flat = u.reshape((M,Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T
  u_diff = u_diff_flat.reshape(bcast_shape + (Nt,Nx))
  u_diff[...,mask] = np.nan
  return u_diff


def space_diff(t,x,u,diffs,mask=None,cuts=None,**kwargs):
  ''' 
  differentiates u
  
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    u : (...,Nt,Nx) array 

    mask : (Nt,Nx) array
      Identifies which elements of u to ignore. This is incase there 
      are outliers or missing data. The returned diffentiated array 
      will have np.nan where the mask is True
      
    kwargs : dict
      passed to space_diff_matrix
      
  Returns
  -------
    u_diff : (Nt,Nx) array
    
  '''
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  u = np.asarray(u,dtype=float)
  Nt,Nx = t.shape[0],x.shape[0]
  if mask is None:
    mask = np.zeros((Nt,Nx),dtype=bool)
  else:
    mask = np.asarray(mask,dtype=bool)
        
  bcast_shape = u.shape[:-2]
  M = np.prod(bcast_shape)

  D = space_diff_matrix(t,x,diffs,mask=mask,cuts=cuts,**kwargs)
  u_flat = u.reshape((M,Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T
  u_diff = u_diff_flat.reshape(bcast_shape + (Nt,Nx))
  u_diff[...,mask] = np.nan
  return u_diff

