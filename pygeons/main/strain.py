''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import logging
import rbf.gauss
from pygeons.main.gptools import composite
from pygeons.main import gpnetwork
from pygeons.main import gpstation
logger = logging.getLogger(__name__)


def _station_sigma_and_p(gp,time,mask):
  ''' 
  Build the covariance matrix and basis vectors describing noise for
  all stations. The covariance and basis functions will only be
  evauluated at unmasked data.
  '''
  # stations that *will* have basis vectors
  sigma_i = gp.covariance(time,time)
  p_i = gp.basis(time)
  _,Nx = mask.shape
  _,Np = p_i.shape
  
  sigma = np.zeros((np.sum(~mask),np.sum(~mask)))
  p = np.zeros((np.sum(~mask),Np,Nx))

  r,c = np.nonzero(~mask)
  for i in range(Nx):
    # good luck trying to figure out what the fuck im doing here
    sigma[np.ix_(c==i,c==i)] = sigma_i[np.ix_(r[c==i],r[c==i])]
    p[c==i,:,i] = p_i[r[c==i]]

  p = p.reshape((np.sum(~mask),Np*Nx))
  if p.size != 0:
    # remove singluar values from p
    u,s,_ = np.linalg.svd(p,full_matrices=False)
    keep = s > 1e-12*s.max()
    p = u[:,keep]
    logger.debug('Removed %s singular values from the station basis vectors' 
                 % np.sum(~keep))
  
  return sigma,p


def strain(t,x,d,sd,
           network_prior_model=('se-se',),
           network_prior_params=(1.0,0.05,50.0),
           network_noise_model=(),
           network_noise_params=(),
           station_noise_model=('p0',),
           station_noise_params=(),
           out_t=None,
           out_x=None):
  ''' 
  Computes deformation gradients from displacement data.
  
  Parameters
  ----------
  t : (Nt,1) array
  x : (Nx,2) array
  d : (Nt,Nx) array
  sd : (Nt,Nx) array
  network_prior_model : str array
  network_prior_params : float array
  network_noise_model : str array
  network_noise_params : float array
  station_noise_model : str array
  station_noise_params : float array
  out_t : (Mt,) array, optional
  out_x : (Mx,2) array, optional
    
  Returns
  -------
  de: (Nt,Nx) array
    edited data
  sde: (Nt,Nx) array
    edited data uncertainty
  fit: (Nt,Nx) float array
  dx: (Mt,Mx) array
  sdx: (Mt,Mx) array
  dy: (Mt,Mx) array
  sdy: (Mt,Mx) array
  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float)
  sd = np.array(sd,dtype=float)
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
  # rebuild noise covariance and basis vectors
  noise_sigma,noise_p = _station_sigma_and_p(sta_gp,t,mask)
  noise_sigma += noise_gp.covariance(z,z)
  noise_p = np.hstack((noise_p,noise_gp.basis(z)))
  rbf.gauss._diag_add(noise_sigma,sd**2)
  
  # condition the prior with the data
  post_gp = prior_gp.condition(z,d,sigma=noise_sigma,p=noise_p)
  dx_gp = post_gp.differentiate((1,1,0)) # x derivative of velocity
  dy_gp = post_gp.differentiate((1,0,1)) # y derivative of velocity

  dx,sdx = dx_gp.meansd(out_z)
  dx = dx.reshape((out_t.shape[0],out_x.shape[0]))
  sdx = sdx.reshape((out_t.shape[0],out_x.shape[0]))

  dy,sdy = dy_gp.meansd(out_z)
  dy = dy.reshape((out_t.shape[0],out_x.shape[0]))
  sdy = sdy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (dx,sdx,dy,sdy)
  return out
