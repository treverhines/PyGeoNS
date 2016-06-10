#!/usr/bin/env python
import numpy as np
from pygeons.smooth import network_smoother
import matplotlib.pyplot as plt
import pygeons.diff
from scipy.spatial import cKDTree
import logging
logger = logging.getLogger(__name__)

def outliers(u,t,x,sigma=None,time_cuts=None,penalty=None,tol=3.0,plot=True):
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

    sigma : (Nt,Nx) array, optional
    
    penalty : float, optional
    
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
    ri,ci = _outliers(u,t,x,sigma=sigma,
                      time_cuts=time_cuts,penalty=penalty,
                      tol=tol,plot=plot)
    logger.info('removed %s outliers on iteration %s' % (ri.shape[0],itr))
    if ri.shape[0] == 0:
      break

    # mark the points as outliers before the next iteration
    sigma[ri,ci] = np.inf
    rout = np.concatenate((rout,ri))      
    cout = np.concatenate((cout,ci))      
    itr += 1
        
  return rout,cout  
  
def _outliers(u,t,x,sigma=None,time_cuts=None,penalty=None,tol=3.0,plot=True):
  ''' 
  single iteration of outliers 
    
  '''
  zero_tol = 1e-10
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  Nt,Nx = u.shape
    
  if penalty is not None:
    penalty = [penalty]
    
  ds = pygeons.diff.make_acceleration_diff_specs()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,sigma=sigma,
                  diff_specs=[ds],
                  penalties=penalty,
                  perts=0,
                  solve_ksp='preonly',
                  solve_pc='lu')
  res = u - upred
  res[np.abs(res) < zero_tol] = 0.0
  if sigma is None:
    sigma = np.ones((Nt,Nx))

  # normalize data by weight
  res /= sigma

  # compute standard deviation of weighted residuals for each station 
  # and ignore missing data marked with sigma=np.inf. 
  res[sigma==np.inf] = np.nan
  std = np.nanstd(res,axis=0)[None,:].repeat(Nt,axis=0)
  res[sigma==np.inf] = 0.0
    
  # identify the largest outlier for each station if one exists
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

def common_mode(u,t,x,sigma=None,time_cuts=None,penalty=0.1,plot=True):  
  ''' 
  returns common mode time series
  
  Parameters
  ----------
    u : (Nt,Nx) array

    sigma : (Nt,Nx) array
    
    penalty : float, optional
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  Nt,Nx = u.shape
    
  if penalty is not None:
    penalty = [penalty]
    
  ds = pygeons.diff.make_acceleration_diff_specs()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,sigma=sigma,
                  diff_specs=[ds],
                  penalties=penalty,
                  perts=0,
                  solve_ksp='preonly',
                  solve_pc='lu')
  res = u - upred
  comm = np.mean(res,axis=1)
  std = np.std(res,axis=1)
  if plot:
    fig,ax = plt.subplots()
    ax.errorbar(t,comm,std,fmt='ro',capsize=0.0,zorder=1)
    ax.plot(t,res,'k.',zorder=0)
    ax.set_xlabel('time')
    ax.set_ylabel('residual')
    ax.legend(['common mode','residuals'],frameon=False)
    fig.tight_layout()
    plt.show()  
                          
  return comm[:,None]
  
  
def duplicates(pos):
  ''' 
  identifies stations which are abnormally close to eachother
  '''
  # if there is zero or one station then dont run this check
  if pos.shape[0] <= 1:
    return

  T = cKDTree(pos)
  dist,idx = T.query(pos,2)
  r = dist[:,1]
  ri = idx[:,1]
  logr = np.log10(r)
  cutoff = np.mean(logr) - 4*np.std(logr)
  duplicates = np.nonzero(logr < cutoff)[0]
  for d in duplicates:
    print('station %s is close to station %s. '
          'This may result in numerical instability. One '
          'of the stations should be removed or they should '
          'be merged together.' % (d,ri[d]))

