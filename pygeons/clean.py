#!/usr/bin/env python
from __future__ import division
import numpy as np
from pygeons.smooth import network_smoother
import matplotlib.pyplot as plt
import pygeons.diff
import pygeons.cuts
from scipy.spatial import cKDTree
import logging
import warnings
logger = logging.getLogger(__name__)


def most(a,axis=None,cutoff=0.5):
  ''' 
  behaves like np.all but returns True if the percentage of Trues
  is greater than or equal to cutoff
  '''
  a = np.asarray(a,dtype=bool)
  if axis is None:
    b = np.prod(a.shape)
  else:
    b = a.shape[axis]

  asum = np.sum(a,axis=axis)
  return asum >= b*cutoff


def outliers(u,t,x,sigma=None,
             time_scale=None,time_cuts=None,tol=3.0,
             plot=True,**kwargs):
  ''' 
  returns indices of time series outliers
  
  An observation is an outlier if the residual between an observation 
  and a smoothed prediction to that observation exceeds tol times the 
  standard deviation of all residuals for the station.  If sigma is 
  given then the residuals are first weighted by sigma. This function 
  removes outliers iteratively, where smoothed curves and residual 
  standard deviations are recomputed after each outlier is removed.
  
  Parameters
  ----------
    u : (Nt,Nx) array

    t : (Nt,) array
    
    x : (Nx,2) array
      this is only used when time_cuts has spatial dependence
      
    sigma : (Nt,Nx) array, optional
    
    time_scale : float, optional
    
    time_cuts : TimeCuts instance, optional
    
    tol : float, optional
    
    plot : bool, optional
    
  Returns
  -------
    row_idx : row indices of outliers 
    
    col_idx : column indices of outliers

  Note
  ----
    masked data needs to be indicated by setting sigma to np.inf.  
    Values where sigma is set to np.inf are explicitly ignored when 
    computing the residual standard deviation. Simply setting sigma 
    equal to a very large number will produce incorrect results.
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
    
  if not (tol > 0.0):
    raise ValueError('tol must be greater than zero')
    
  if sigma is None:
    sigma = np.ones(u.shape)
  else:
    # elements in sigma will be changed and so a copy is needed
    sigma = np.array(sigma,copy=True)
    
  rout = np.zeros((0,),dtype=int)
  cout = np.zeros((0,),dtype=int)
  itr = 0  
  while True:
    ri,ci = _outliers(u,t,x,sigma,
                      time_scale,time_cuts,tol,
                      plot,**kwargs)
    logger.info('removed %s outliers on iteration %s' % (ri.shape[0],itr))
    if ri.shape[0] == 0:
      break

    # mark the points as outliers before the next iteration
    sigma[ri,ci] = np.inf
    rout = np.concatenate((rout,ri))      
    cout = np.concatenate((cout,ci))      
    itr += 1
        
  return rout,cout  
  

def _outliers(u,t,x,sigma,
              time_scale,time_cuts,tol,
              plot,**kwargs):
  ''' 
  single iteration of outliers 
  '''
  zero_tol = 1e-10
  Nt,Nx = u.shape
    
  ds = pygeons.diff.acc()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,sigma=sigma,
                  diff_specs=[ds],
                  time_scale=time_scale,
                  perts=0,**kwargs)
  res = u - upred
  # it is possible that there are not enough observations to make the 
  # problem overdetermined. In such case the solution should be exact. 
  # If the residual is less than zero tol then we can assume the 
  # solution is supposed to be exact and any errors are due to 
  # numerical rounding. 
  res[np.abs(res) < zero_tol] = 0.0
  if sigma is None:
    sigma = np.ones((Nt,Nx))

  # normalize data by weight
  res /= sigma

  # compute standard deviation of weighted residuals for each station 
  # and ignore missing data marked with sigma=np.inf. 
  res[np.isinf(sigma)] = np.nan
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    std = np.nanstd(res,axis=0)[None,:].repeat(Nt,axis=0)
    # if there are too few degrees of freedom then make std 0
    std[np.isnan(std)] = 0.0
    
  res[np.isinf(sigma)] = 0.0
  # remove all outliers 
  absres = np.abs(res)
  idx_row,idx_col = np.nonzero((absres > tol*std) & 
                               (absres == np.max(absres,axis=0)))

  if plot:
    # plot each time series where an outlier as been identified
    for r,c in zip(idx_row,idx_col):
      fig,ax = plt.subplots(2,1,sharex=True)      
      has_finite_sigma, = np.nonzero(~np.isinf(sigma[:,c]))
      ax[0].plot(t[has_finite_sigma],
                 u[has_finite_sigma,c],
                 'k.')
      ax[0].plot(t[has_finite_sigma],
                 upred[has_finite_sigma,c],'b-')  
      ax[0].plot(t[r],u[r,c],'ro')
      ax[0].grid() 
      ax[0].set_xlabel('time')
      ax[0].set_ylabel('displacement')
      ax[0].set_title('outliers detected for station %s' % c)
      ax[0].legend(['observed','predicted','outliers'],frameon=False)

      ax[1].fill_between(t,-tol*std[:,c],tol*std[:,c],color='b',alpha=0.2)  
      ax[1].plot(t[has_finite_sigma],
                 res[has_finite_sigma,c],'k.')  
      ax[1].plot(t[r],res[r,c],'ro')
      ax[1].plot(t,0*t,'b-')
      ax[1].grid() 
      ax[1].set_xlabel('time')
      ax[1].set_ylabel('weighted residual')
      fig.tight_layout()
      plt.show()        
      
  return idx_row,idx_col


def common_mode(u,t,x,sigma=None,
               time_scale=None,time_cuts=None,
               plot=True,**kwargs):  
  ''' 
  returns common mode time series. Common mode is a weighted mean 
  residual time series between all stations
  
  Parameters
  ----------
    u : (Nt,Nx) array

    t : (Nt,) array
    
    x : (Nx,2) array
    
    sigma : (Nt,Nx) array, optional
    
    time_scale : float, optional
    
    time_cuts : TimeCuts instance, optional
    
    plot : bool, optional

  Returns
  -------
    u_comm : (Nt,1) array

    sigma_comm : (Nt,1) array
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  if sigma is None:
    sigma = np.ones(u.shape)
  else:
    sigma = np.asarray(sigma)

  Nt,Nx = u.shape
    
  ds = pygeons.diff.acc()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,sigma=sigma,
                  diff_specs=[ds],
                  time_scale=time_scale,
                  perts=0,**kwargs)
  res = u - upred
    
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    numer = np.sum(res/sigma**2,axis=1)
    denom = np.sum(1.0/sigma**2,axis=1)
    u_comm = numer/denom
    sigma_comm = np.sqrt(1.0/denom)

  #comm = np.ma.average(res,axis=1,weights=1.0/sigma)
  # if a time has no observations then make its common mode zero
  u_comm[np.isnan(u_comm)] = 0.0
  if plot:
    fig,ax = plt.subplots()
    masked_res = np.ma.masked_array(res,mask=np.isinf(sigma))
    ax.plot(t,u_comm,'ro',zorder=1)
    ax.plot(t,masked_res,'k.',zorder=0)
    ax.set_xlabel('time')
    ax.set_ylabel('residual')
    ax.legend(['common mode','residuals'],frameon=False)
    fig.tight_layout()
    plt.show()  
                          
  u_comm = u_comm[:,None]
  sigma_comm = sigma_comm[:,None]
  return u_comm,sigma_comm
  

def baseline(u,t,x,sigma=None,time_scale=None,
             zero_idx=None,time_cuts=None,perts=20,**kwargs):
  ''' 
  Estimates the displacements at t_zero for each station. 
  Estimates of displacement are made with a weighted mean of 
  displacements within some time interval of t_zero

  Parameters
  ----------
    u : (Nt,Nx) array
 
    t : (Nt,) array
 
    x : (Nx,2) array

    sigma : (Nt,Nx) array, optional
    
    zero_idx : int, optional

    time_cuts : TimeCuts instance, optional

  Returns 
  -------
    u_out : (Nt,Nx) array

    sigma_out : (Nt,Nx) array
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  Nt,Nx = u.shape
  if sigma is None:
    sigma = np.ones(u.shape)
  else:
    sigma = np.asarray(sigma)

  if zero_idx is None:
    zero_idx = Nt//2
    
  ds = pygeons.diff.acc()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,sigma=sigma,
                  diff_specs=[ds],
                  time_scale=time_scale,
                  perts=perts,**kwargs)
  sigma = np.std(upert,axis=0)                  

  u_out = upred[[zero_idx],:]   
  sigma_out = sigma[[zero_idx],:]
  return u_out,sigma_out


def network_cleaner(u,t,x,sigma=None,time_cuts=None,
                    tol=3.0,zero_idx=None,time_scale=None,plot=True,
                    perts=20,**kwargs):
  ''' 
  Parameters
  ----------
    u : (Nt,Nx) array
    
    t : (Nt,) array

    x : (Nx,2) array
    
    sigma : (Nt,Nx) array
    
    time_cuts : TimeCutCollection 
    
    tol : scalar
    
    time_scale : scalar    
    
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  if sigma is None:
    sigma = np.ones(u.shape)
  else:
    sigma = np.array(sigma,copy=True)  
    
  # identify outliers
  ridx,cidx = outliers(u,t,x,sigma=sigma,time_cuts=time_cuts,
                       tol=tol,time_scale=time_scale,plot=plot,
                       **kwargs)
                       
  sigma[ridx,cidx] = np.inf
  # u[ridx,cidx] = 0.0

  # remove common mode
  u_comm,sigma_comm = common_mode(u,t,x,sigma=sigma,time_cuts=time_cuts,
                        time_scale=time_scale,plot=plot,
                        **kwargs)
  u = u - u_comm                     
  sigma = np.sqrt(sigma**2 + sigma_comm**2)

  # remove baseline
  u_base,sigma_base = baseline(u,t,x,sigma=sigma,time_cuts=time_cuts,
                               time_scale=time_scale,zero_idx=zero_idx,
                               perts=perts)
  u = u - u_base
  sigma = np.sqrt(sigma**2 + sigma_base**2)

  # stations which have anomalously large uncertainties are given inf 
  # uncertainty so that they can be treated as masked data
  mean_sigma = np.mean(sigma[np.isfinite(sigma)])
  std_sigma = np.std(sigma[np.isfinite(sigma)])
  sigma[sigma > (mean_sigma + tol*std_sigma)] = np.inf

  return u,sigma
                     

def time_lacks_data(sigma,cutoff=0.5):
  ''' 
  returns true for each time where sigma is np.inf for most of the stations.
  The percentage of infs must exceed cutoff in order to return True.
  '''
  out = most(np.isinf(sigma),axis=1,cutoff=cutoff)
  return out


def station_lacks_data(sigma,cutoff=0.5):
  ''' 
  returns true for each station where sigma is np.inf for most of the times.
  The percentage of infs must exceed cutoff in order to return True.
  '''
  out = most(np.isinf(sigma),axis=0,cutoff=cutoff)
  return out


def station_is_duplicate(sigma,x,tol=4.0,plot=True):
  ''' 
  returns the indices for a set of nonduplicate stations. if duplicate 
  stations are found then the one that has the most observations is 
  retained
  '''
  x = np.asarray(x)
  # if there is zero or one station then dont run this check

  is_duplicate = np.zeros(x.shape[0],dtype=bool)
  while True:
    xi = x[~is_duplicate]
    sigmai = sigma[:,~is_duplicate]
    idx = _identify_duplicate_station(sigmai,xi,tol=tol) 
    if idx is None:
      break
    else:  
      global_idx = np.nonzero(~is_duplicate)[0][idx]
      logger.info('identified station %s as a duplicate' % global_idx)
      is_duplicate[global_idx] = True

  if plot:
    # get nearest neighbors
    T = cKDTree(x)
    dist,idx = T.query(x,2)
    rbefore = dist[:,1]
    T = cKDTree(x[~is_duplicate])
    dist,idx = T.query(x[~is_duplicate],2)
    rafter = dist[:,1]
    fig,ax = plt.subplots()
    bin_count = max(len(x)//10,10)
    out,bins,patches = ax.hist(np.log10(rbefore),bin_count,color='r',edgecolor='none')
    ax.hist(np.log10(rafter),bins,color='k',edgecolor='none')
    ax.set_xlabel('log10 distance to nearest neighbor')
    ax.set_ylabel('count')
    ax.legend(['outliers'],loc=2,frameon=False)
    ax.grid()
    plt.show()
    
  return is_duplicate      


def _identify_duplicate_station(sigma,x,tol=3.0):
  ''' 
  returns the index of the station which is likely to be a duplicate 
  due to its proximity to another station.  The station which has the 
  least amount of data is identified as the duplicate
  '''
  # if there is zero or one station then dont run this check
  if x.shape[0] <= 1:
    return None
    
  T = cKDTree(x)
  dist,idx = T.query(x,2)
  r = dist[:,1]
  ri = idx[:,1]
  logr = np.log10(r)
  cutoff = np.mean(logr) - tol*np.std(logr)
  if not np.any(logr < cutoff):
    # if no duplicates were found then return nothing
    return None

  else:
    # otherwise return the index of the least useful of the two 
    # nearest stations
    idx1 = np.argmin(logr)    
    idx2 = ri[idx1]
    count1 = np.sum(~np.isinf(sigma[:,idx1]))
    count2 = np.sum(~np.isinf(sigma[:,idx2]))
    count,out = min((count1,idx1),(count2,idx2))
    return out
  
