''' 
Contains a function for computing strain or strain rates.
'''
import numpy as np
import scipy.sparse as sp
import logging
from rbf.gauss import (_as_sparse_or_array,
                       _as_array,
                       _as_covariance)
from pygeons.main.gptools import composite
from pygeons.main import gpnetwork
from pygeons.main import gpstation
import warnings
logger = logging.getLogger(__name__)


def _station_sigma_and_p(gp,time,mask):
  ''' 
  Build the sparse covariance matrix and basis vectors describing
  noise for all stations. The covariance and basis functions will only
  be evauluated at unmasked data.
  '''
  logger.debug('Building station covariance matrix and basis vectors ...')
  diff = np.array([0])
  sigma_i = gp._covariance(time,time,diff,diff)
  # convert sigma_i to a dense array, even if it is sparse
  sigma_i = _as_array(sigma_i)
  p_i = gp._basis(time,diff)
  _,Np = p_i.shape
  Nt,Nx = mask.shape

  # build the sparse covariance matrix for all data (including the
  # masked ones) then crop out the masked ones afterwards. This is
  # not efficient but I cannot think of a better way.
  sigma = sp.csc_matrix((Nt*Nx,Nt*Nx))
  p = np.zeros((Nt*Nx,Np*Nx))
  # scipy will warn you about sparse inefficiencies but this really is
  # the most memory efficient option that I can find
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(Nx):
      sigma[i::Nx,i::Nx] = sigma_i 
      p[i::Nx,i::Nx] = p_i
    
  # toss out masked rows and columns
  maskf = mask.ravel()
  sigma = sigma[:,~maskf][~maskf,:]
  p = p[~maskf,:]
  logger.debug('Station covariance matrix is sparse with %.3f%% non-zeros' % 
               (sigma.nnz/(1.0*np.prod(sigma.shape))))
  
  if p.size != 0:
    # remove singluar values from p
    u,s,_ = np.linalg.svd(p,full_matrices=False)
    keep = s > 1e-12*s.max()
    p = u[:,keep]
    logger.debug('Removed %s singular values from the station basis vectors' 
                 % np.sum(~keep))
  
  logger.debug('Done')
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
  # build noise covariance and basis vectors
  sta_sigma,sta_p = _station_sigma_and_p(sta_gp,t,mask)
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

  u,su = u_gp.meansd(out_z,chunk_size=1000)
  u = u.reshape((out_t.shape[0],out_x.shape[0]))
  su = su.reshape((out_t.shape[0],out_x.shape[0]))

  dudx,sdudx = dudx_gp.meansd(out_z,chunk_size=1000)
  dudx = dudx.reshape((out_t.shape[0],out_x.shape[0]))
  sdudx = sdudx.reshape((out_t.shape[0],out_x.shape[0]))

  dudy,sdudy = dudy_gp.meansd(out_z,chunk_size=1000)
  dudy = dudy.reshape((out_t.shape[0],out_x.shape[0]))
  sdudy = sdudy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (u,su,dudx,sdudx,dudy,sdudy)
  return out
