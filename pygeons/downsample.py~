#!/usr/bin/env python 
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree
import rbf.geometry
import warnings
import pygeons.cuts

def weighted_mean(x,sigma,axis=None):
  ''' 
  computes the weighted mean of *x* with uncertainties *sigma*
  
  Parameters
  ----------
    x : (..., N,...) array
    
    sigma : (..., N,...) array
      
    axis : int, optional
    
  Notes
  -----
    If all uncertainties along the axis are np.inf then then the 
    returned mean is np.nan with uncertainty is np.inf
    
    If there are 0 entries along the axis then the returned mean is 
    np.nan with uncertainty np.inf
    
    zero uncertainties will raise an error
    
  '''
  x = np.array(x,copy=True)
  # convert any nans to zeros
  x[np.isnan(x)] = 0.0
  sigma = np.asarray(sigma)

  if x.shape != sigma.shape:
    raise ValueError('x and sigma must have the same shape')

  # make sure uncertainties are positive
  if np.any(sigma <= 0.0):
    raise ValueError('uncertainties must be positive and nonzero') 

  numer = np.sum(x/sigma**2,axis=axis)
  denom = np.sum(1.0/sigma**2,axis=axis)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # out_value can be nan if the arrays have zero length along axis 
    # or if sigma is inf. out_sigma will be inf in that case 
    out_value = numer/denom
    out_sigma = np.sqrt(1.0/denom)
  
  return out_value,out_sigma


class MeanInterpolant:
  '''   
  An interplant whose value at x is the mean of all values observed 
  within some radius of x
  
  If no values are within the radius of a queried position then the
  returned value is np.nan with np.inf for its uncertainty
  
  If all values within a radius have np.inf for their uncertainty
  then the returned value is np.nan with np.inf for its uncertainty
  '''
  def __init__(self,x,value,sigma=None,vert=None,smp=None):
    ''' 
    Parameters
    ----------
      x : (N,D) array

      value : (...,N) array

      sigma : (...,N) array

    '''
    x = np.asarray(x,dtype=float)
    value = np.asarray(value,dtype=float)
    if sigma is None:
      sigma = np.ones(value.shape,dtype=float)
    else:
      sigma = np.asarray(sigma,dtype=float)

    if vert is None:
      vert = np.zeros((0,x.shape[1]),dtype=float)
    else:
      vert = np.asarray(vert)

    if smp is None:
      smp = np.zeros((0,x.shape[1]),dtype=int)
    else:
      smp = np.asarray(smp)

    # form observation KDTree 
    self.Tobs = cKDTree(x)
    self.x = x
    self.value = value
    self.sigma = sigma
    self.bcast_shape = value.shape[:-1]
    self.vert = vert
    self.smp = smp

  def __call__(self,xitp,radius=None):
    ''' 
    Parameters
    ----------
      x : (K,D) array

      radius : scalar, optional
        values within this distance from the interpolation points are 
        used to compute the mean. If not given then half the average 
        distance between interpolation points is used. If xitp has 
        only one point then radius is inf

      vert : (P,D) array, optional

      smp : (P,D) int array, optional
    Returns
    -------  
      out_value : (...,K) array

      out_sigma : (...,K) array
    '''
    xitp = np.asarray(xitp)
    Nitp = xitp.shape[0]
    Titp = cKDTree(xitp)
    if radius is None:
      # half of the average nearest neighbor distance
      radius = np.mean(Titp.query(xitp,2)[0][:,1])/2.0

    idx_arr = Titp.query_ball_tree(self.Tobs,radius)
    # make sure that the line segments connecting the observation 
    # points to the interpolation point do not intersect a boundary
    if self.smp.shape[0] != 0:
      for i in range(Nitp):
        idx = idx_arr[i]
        xitp_ext = np.repeat(xitp[[i]],len(idx),axis=0)
        count = rbf.geometry.intersection_count(self.x[idx],xitp_ext,
                                                self.vert,self.smp)
        # throw out observation points that cross a boundary
        idx_arr[i] = [j for j,c in zip(idx,count) if c == 0]
        
    out_value = np.zeros(self.bcast_shape + (xitp.shape[0],))
    out_sigma = np.zeros(self.bcast_shape + (xitp.shape[0],))
    for i,idx in enumerate(idx_arr):
      out_value[...,i],out_sigma[...,i] = weighted_mean(
                                            self.value[...,idx],
                                            self.sigma[...,idx],
                                            axis=-1)
    
    return out_value,out_sigma


def downsample(t,tnew,x,u,sigma=None,cuts=None):
  ''' 
  Downsamples the data from times *t* to *tnew*. This is done by 
  applying a weighted mean over each new time interval. The weighted 
  mean does not run over any specified time cuts
  
  Parameters
  ---------- 
    t : (Nt,) array
    
    tnew : (Nitp,) array

    x : (Nx,2) array

    u : (...,Nt,Nx) array

    sigma : (Nt,Nx) array, optional
    
    cuts : (Nc,) array

  Returns
  -------
    u_out : (...,Nitp,Nx) array

    sigma_out :(Nitp,Nx) array
  
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  tnew = np.asarray(tnew)
  x = np.asarray(x)
  if cuts is None:
    vert = None
    smp = None
  else:
    vert = np.array(cuts).reshape((-1,1))
    smp = np.arange(vert.shape[0]).reshape((-1,1))

  if sigma is None:
    sigma = np.ones(u.shape)
  else:
    # broadcast sigma to the shape of u
    a = np.ones(u.shape)
    sigma = a*sigma
    
  Nitp = tnew.shape[0]
  Nt = t.shape[0]
  Nx = x.shape[0]
  bcast_shape = u.shape[:-2]
  
  # reshape u so that time is the last axis
  u = np.einsum('...ij->...ji',u)
  sigma = np.einsum('...ij->...ji',sigma)
  I = MeanInterpolant(t[:,None],u,sigma=sigma,vert=vert,smp=smp)
  u_out,sigma_out = I(tnew[:,None])

  # reshape the output so that space is the last axis
  u_out = np.einsum('...ij->...ji',u_out)
  sigma_out = np.einsum('...ij->...ji',sigma_out)
  
  # return just the first sigma_out
  sigma_out = sigma_out.reshape((-1,Nitp,Nx))[0]
  return u_out,sigma_out











