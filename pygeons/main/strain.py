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
           network_prior_model,
           network_prior_params,
           network_noise_model,
           network_noise_params,
           station_noise_model,
           station_noise_params,
           out_t,out_x,rate):
  ''' 
  Computes deformation gradients from displacement data.
  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float)
  sd = np.array(sd,dtype=float)
  diff = np.array([0,0,0])

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

  prior_gp = composite(network_prior_model,network_prior_params,gpnetwork.CONSTRUCTORS)
  noise_gp = composite(network_noise_model,network_noise_params,gpnetwork.CONSTRUCTORS)
  sta_gp   = composite(station_noise_model,station_noise_params,gpstation.CONSTRUCTORS)

  # find missing data
  mask = np.isinf(sd)
  # get unmasked data and uncertainties
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
    dudx_gp = post_gp.differentiate((1,1,0)) # x derivative of velocity
    dudy_gp = post_gp.differentiate((1,0,1)) # y derivative of velocity

  else:  
    dudx_gp = post_gp.differentiate((0,1,0)) # x derivative of displacement
    dudy_gp = post_gp.differentiate((0,0,1)) # y derivative of displacement

  dudx,sdudx = dudx_gp.meansd(out_z,chunk_size=1000)
  dudy,sdudy = dudy_gp.meansd(out_z,chunk_size=1000)
        
  dudx = dudx.reshape((out_t.shape[0],out_x.shape[0]))
  sdudx = sdudx.reshape((out_t.shape[0],out_x.shape[0]))
  dudy = dudy.reshape((out_t.shape[0],out_x.shape[0]))
  sdudy = sdudy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (dudx,sdudx,dudy,sdudy)
  return out
