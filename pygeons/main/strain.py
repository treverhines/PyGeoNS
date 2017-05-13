''' 
Contains a function for computing strain or strain rates.
'''
import numpy as np
import logging
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from rbf.gauss import (_as_sparse_or_array,
                       _as_covariance)
from pygeons.main.gptools import (composite,
                                  station_sigma_and_p)

logger = logging.getLogger(__name__)


def strain(t,x,d,sd,
           network_prior_model=('se-se',),
           network_prior_params=(5.0,0.05,50.0),
           network_noise_model=(),
           network_noise_params=(),
           station_noise_model=('p0','p1'),
           station_noise_params=(),
           out_t=None,
           out_x=None,
           rate=True,
           uncertainty=False):
  ''' 
  Computes deformation gradients from displacement data.
  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float)
  sd = np.array(sd,dtype=float)
  diff = np.array([0,0,0])
  # allocate array indicating which data have been removed
  if out_t is None:
    out_t = t

  if out_x is None:
    out_x = x

  prior_gp = composite(network_prior_model,network_prior_params,gpnetwork.CONSTRUCTORS)
  noise_gp = composite(network_noise_model,network_noise_params,gpnetwork.CONSTRUCTORS)
  sta_gp   = composite(station_noise_model,station_noise_params,gpstation.CONSTRUCTORS)

  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')  
  # flat observation times and positions
  z = np.array([t_grid.ravel(),
                x0_grid.ravel(),
                x1_grid.ravel()]).T

  t_grid,x0_grid = np.meshgrid(out_t,out_x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(out_t,out_x[:,1],indexing='ij')  
  # flat observation times and positions
  out_z = np.array([t_grid.ravel(),
                    x0_grid.ravel(),
                    x1_grid.ravel()]).T

  # find missing data
  mask = np.isinf(sd)
  # unmasked data and uncertainties
  z,d,sd = z[~mask.ravel()],d[~mask],sd[~mask]
  # build noise covariance and basis vectors
  sta_sigma,sta_p = station_sigma_and_p(sta_gp,t,mask)
  # add data noise to the station noise
  obs_sigma = _as_covariance(sd)
  sta_sigma = _as_sparse_or_array(sta_sigma + obs_sigma)
  # make network noise
  net_sigma = noise_gp._covariance(z,z,diff,diff)
  net_p = noise_gp._basis(z,diff)
  # combine noise processes
  noise_sigma = _as_sparse_or_array(sta_sigma + net_sigma)
  noise_p = np.hstack((sta_p,net_p))
  del sta_sigma,net_sigma,obs_sigma,sta_p,net_p
  # condition the prior with the data
  post_gp = prior_gp.condition(z,d,sigma=noise_sigma,p=noise_p)
  if rate:
    u_gp    = post_gp.differentiate((1,0,0)) # velocity
    dudx_gp = post_gp.differentiate((1,1,0)) # x derivative of velocity
    dudy_gp = post_gp.differentiate((1,0,1)) # y derivative of velocity

  else:  
    u_gp    = post_gp.differentiate((0,0,0)) # displacement
    dudx_gp = post_gp.differentiate((0,1,0)) # x derivative of displacement
    dudy_gp = post_gp.differentiate((0,0,1)) # y derivative of displacement

  if uncertainty:
    # compute the uncertainties, which can be very expensive
    u,su = u_gp.meansd(out_z,chunk_size=1000)
    dudx,sdudx = dudx_gp.meansd(out_z,chunk_size=1000)
    dudy,sdudy = dudy_gp.meansd(out_z,chunk_size=1000)

  else:
    # return zeros for the uncertainties
    u = u_gp.mean(out_z)
    su = np.zeros_like(u)
    dudx = dudx_gp.mean(out_z)
    sdudx = np.zeros_like(u)
    dudy = dudy_gp.mean(out_z)
    sdudy = np.zeros_like(u)
        
  u = u.reshape((out_t.shape[0],out_x.shape[0]))
  su = su.reshape((out_t.shape[0],out_x.shape[0]))
  dudx = dudx.reshape((out_t.shape[0],out_x.shape[0]))
  sdudx = sdudx.reshape((out_t.shape[0],out_x.shape[0]))
  dudy = dudy.reshape((out_t.shape[0],out_x.shape[0]))
  sdudy = sdudy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (u,su,dudx,sdudx,dudy,sdudy)
  return out
