''' 
Contains a function for computing strain or strain rates.
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
  diff = np.array([0])
  # use _covariance and _basis instead of covariance and basis because
  # they do not make copies
  sigma_i = gp._covariance(time,time,diff,diff)
  p_i = gp._basis(time,diff)
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
           network_prior_params=(5.0,0.05,50.0),
           network_noise_model=(),
           network_noise_params=(),
           station_noise_model=('p0','p1'),
           station_noise_params=(),
           out_t=None,
           out_x=None,
           rate=True):
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
  # rebuild noise covariance and basis vectors
  noise_sigma,noise_p = _station_sigma_and_p(sta_gp,t,mask)
  # use _covariance and _basis instead of covariance and basis because
  # they do not make copies
  noise_sigma += noise_gp._covariance(z,z,diff,diff)
  noise_p = np.hstack((noise_p,noise_gp._basis(z,diff)))
  rbf.gauss._diag_add(noise_sigma,sd**2)
  
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

  u,su = u_gp.meansd(out_z,chunk_size=500)
  u = u.reshape((out_t.shape[0],out_x.shape[0]))
  su = su.reshape((out_t.shape[0],out_x.shape[0]))

  dudx,sdudx = dudx_gp.meansd(out_z,chunk_size=500)
  dudx = dudx.reshape((out_t.shape[0],out_x.shape[0]))
  sdudx = sdudx.reshape((out_t.shape[0],out_x.shape[0]))

  dudy,sdudy = dudy_gp.meansd(out_z,chunk_size=500)
  dudy = dudy.reshape((out_t.shape[0],out_x.shape[0]))
  sdudy = sdudy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (u,su,dudx,sdudx,dudy,sdudy)
  return out
