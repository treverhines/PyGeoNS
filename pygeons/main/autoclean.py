''' 
Contains a data editing algorithm.
'''
import numpy as np
import logging
from pygeons.main.gptools import composite
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from pygeons.main.strain import _station_sigma_and_p
from rbf.gauss import (_as_sparse_or_array,
                       outliers)
logger = logging.getLogger(__name__)


def autoclean(t,x,d,sd,
              network_model=('se-se',),
              network_params=(5.0,0.05,50.0),
              station_model=('p0','p1'),
              station_params=(),tol=4.0):
  ''' 
  Returns a dataset that has been cleaned of outliers using a data
  editing algorithm.

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
  tol : float

  '''
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  de = np.array(d,dtype=float,copy=True)
  sde = np.array(sd,dtype=float,copy=True)
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
  mask = np.isinf(sde)
  zu,du,sdu = z[~mask.ravel()],de[~mask],sde[~mask]
  # Build covariance and basis vectors for the combined process. Do
  # not evaluated at masked points
  sta_sigma,sta_p = _station_sigma_and_p(sta_gp,t,mask)
  net_sigma = net_gp._covariance(zu,zu,diff,diff)
  net_p = net_gp._basis(zu,diff)
  # combine station gp with the network gp
  mu = np.zeros(z.shape[0])  
  sigma = _as_sparse_or_array(sta_sigma + net_sigma)
  p = np.hstack((sta_p,net_p))
  del sta_sigma,net_sigma,sta_p,net_p
  # returns the indices of outliers 
  out_idx = outliers(du,sdu,
                     mu=mu,sigma=sigma,p=p,
                     tol=tol)
  # mask the outliers in *de* and *sde*
  r,c = np.nonzero(~mask)
  de[r[out_idx],c[out_idx]] = np.nan
  sde[r[out_idx],c[out_idx]] = np.inf
  return (de,sde)
