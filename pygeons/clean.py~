#!/usr/bin/env python
import numpy as np
from pygeons.smooth import network_smoother
import matplotlib.pyplot as plt
import pygeons.diff

def outliers(u,t,x,sigma=None,time_cuts=None,alpha=None,tol=3.0,plot=True):
  ''' 
  returns indices of time series outliers
  
  Parameters
  ----------
    u : (Nt,Nx) array

    sigma : (Nt,Nx) array, optional
    
    alpha : float, optional
  '''
  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)
  Nt,Nx = u.shape

  if Nt < 3:
    return np.zeros(0),np.zeros(0)
    
  if alpha is not None:
    alpha = [alpha]
    
  ds = pygeons.diff.make_acceleration_diff_specs()
  ds['time']['cuts'] = time_cuts
  upred,upert = network_smoother(
                  u,t,x,
                  diff_specs=[ds],
                  penalties=alpha,
                  perts=0,
                  solve_ksp='preonly',
                  solve_pc='lu')
  res = u - upred
  std = np.std(res)
  residual_exceeds_tol = np.abs(res) > tol*std
  if sigma is not None:
    sigma_exceeds_tol = sigma > tol*std 
  else:
    sigma_exceeds_tol = np.zeros((Nt,Nx),dtype=bool)
    
  idx_row,idx_col = np.nonzero(residual_exceeds_tol | sigma_exceeds_tol)
  if plot:
    # plot each time series where an outlier as been identified
    unique_cols = np.unique(idx_col)
    for uc in unique_cols:
      i = np.nonzero(idx_col==uc)[0]
      r = idx_row[i]
      c = idx_col[i]
      
      fig,ax = plt.subplots(2,1)      
      ax[0].fill_between(t,upred[:,uc]-tol*std,upred[:,uc]+tol*std,color='b',alpha=0.2)  
      if sigma is None:
        ax[0].plot(t,u[:,uc],'k.')  
      else:  
        ax[0].errorbar(t,u[:,uc],sigma[:,uc],fmt='k.',capsize=0.0)  

      ax[0].plot(t,upred[:,uc],'b-')  
      ax[0].plot(t[r],u[r,c],'ro')
      ax[0].grid() 
      ax[0].set_xlabel('time')
      ax[0].set_ylabel('displacement')
      ax[0].set_title('outliers detected for station %s' % uc)
      ax[0].legend(['observed','predicted','outliers'],frameon=False)

      ax[1].fill_between(t,-tol*std,tol*std,color='b',alpha=0.2)  
      if sigma is None:
        ax[1].plot(t,res[:,uc],'k.')  
      else:  
        ax[1].errorbar(t,res[:,uc],sigma[:,uc],fmt='k.',capsize=0.0)  
      
      ax[1].plot(t[r],res[r,c],'ro')
      ax[1].plot(t,0*t,'b-')
      ax[1].grid() 
      ax[1].set_xlabel('time')
      ax[1].set_ylabel('residual')
      fig.tight_layout()
      plt.show()        
      
  return idx_row,idx_col
  
def common_mode(u,t,x,sigma=None,alpha=0.1):  
  ''' 
  returns common mode time series
  
  Parameters
  ----------
    u : (Nt,Nx) array

    sigma : (Nt,Nx) array
    
    alpha : float, optional
  '''
  return comm
  
def duplicates(lons,lats):  
  ''' 
  returns list of duplicate station indices
  '''
  return idx

  
  
  
