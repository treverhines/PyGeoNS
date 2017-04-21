''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import logging
import rbf.gauss
from pygeons.main.gprocs import gpcomp
logger = logging.getLogger(__name__)


def _is_outlier(d,s,sigma,mu,p,tol):
  ''' 
  Identifies which points in *d* are outliers based on the prior
  defined by *sigma*, *mu*, and *p*.
  
  d : (N,) observations 
  s : (N,) observation uncertainties
  sigma : (N,N) prior covariance matrix
  mu : (N,) prior mean
  p : (N,P) prior basis functions
  tol : outlier tolerance
  
  returns a boolean array indicating which points are outliers and the
  best fit trend to the data.
  
  WARNING: This function has been written to maximize memory
  efficiency. consequently it is really hard to follow.
  '''
  itr = 1
  out = np.zeros(d.shape[0],dtype=bool)
  while True:
    logger.debug('Starting iteration %s of outlier detection routine' % itr)
    mask = np.isinf(s) | out
    q = sum(~mask)
    # K : gp cov. + data cov.
    K = sigma[np.ix_(~mask,~mask)] + np.diag(s[~mask]**2)
    # mat : inverse of gp cov. + data cov. and augmented basis vecs.
    K = rbf.gauss._cholesky_block_inv(K,p[~mask])
    # residual of observed and mean 
    r = np.zeros(q+p.shape[1])
    r[:q] = d[~mask] - mu[~mask]
    # dot residual with inverse of the covariances
    Kr = K.dot(r)
    # form prediction vector 
    fit = mu + sigma[:,~mask].dot(Kr[:q]) + p.dot(Kr[q:])
    res = np.abs(fit - d)/s
    res[np.isinf(s)] = np.inf
    rms = np.sqrt(np.mean(res[~mask]**2))
    if np.all(mask == (res > tol*rms)):
      break

    else:
      out = (res > tol*rms) & ~np.isinf(s)
      itr += 1

  logger.debug('Detected %s outliers out of %s observations' % (sum(out),sum(~np.isinf(s))))
  return out,fit


def _remove_zero_columns(A):
  ''' 
  remove columns of *A* which only have zeros
  '''
  toss = np.all(A == 0.0,axis=0)
  A = A[:,~toss]
  return A


def strain(t,x,d,sd,
           prior_model=('se-se',),
           prior_params=(1.0,0.05,50.0),
           noise_model=('null',),
           noise_params=(),
           station_noise_model=('p0',),
           station_noise_params=(),
           out_t=None,
           out_x=None,
           tol=4.0):
  ''' 
  Computes deformation gradients from displacement data.
  
  Parameters
  ----------
  t : (Nt,1) array
  x : (Nx,2) array
  d : (Nt,Nx) array
  sd : (Nt,Nx) array
  prior_model : str array
  prior_params : float array
  noise_model : str array
  noise_params : float array
  station_noise_model : str array
  station_noise_params : float array
  out_t : (Mt,) array, optional
  out_x : (Mx,2) array, optional
  tol : float, optional
    
  Returns
  -------
  removed: (Nt,Nx) bool array
  fit: (Nt,Nx) float array
  dx: (Mt,Mx) array
  sigma_dx: (Mt,Mx) array
  dy: (Mt,Mx) array
  sigma_dy: (Mt,Mx) array
  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float,copy=True)
  sd = np.array(sd,dtype=float,copy=True)
  Nt,Nx = t.shape[0],x.shape[0]
  # allocate array indicating which data have been removed
  removed = np.zeros((Nt,Nx),dtype=bool)
  if out_t is None:
    out_t = t

  if out_x is None:
    out_x = x

  prior_gp = gpcomp(prior_model,prior_params)
  noise_gp = gpcomp(noise_model,noise_params)    
  sta_gp   = gpcomp(station_noise_model,station_noise_params)

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

  # create covariance matrix and basis functions describing noise.
  # start with station-specific noise.
  sta_sigma = sta_gp.covariance(t,t)
  sta_p = sta_gp.basis(t)
  Np = sta_p.shape[1]

  # If a station has insufficient data for the station basis vectors
  # then mask whatever few data points it does have 
  bad_stations_bool = np.sum(~np.isinf(sd),axis=0) < Np
  bad_stations,  = np.nonzero(bad_stations_bool)
  good_stations, = np.nonzero(~bad_stations_bool) 
  d[:,bad_stations] = np.nan
  sd[:,bad_stations] = np.inf
  removed[:,bad_stations] = True
  
  # expand so that there is a covariance matrix and basis functions
  # for each station that has sufficient data. 
  noise_sigma = np.zeros((Nt,Nx,Nt,Nx))
  noise_p = np.zeros((Nt,Nx,Np,len(good_stations)))
  for i,j in enumerate(good_stations):
    noise_sigma[:,j,:,j] = sta_sigma
    noise_p[:,j,:,i] = sta_p
    
  noise_sigma = noise_sigma.reshape((Nt*Nx,Nt*Nx))
  noise_p = noise_p.reshape((Nt*Nx,Np*len(good_stations)))

  # add spatial noise covariance and basis
  noise_sigma += noise_gp.covariance(z,z)
  noise_p = np.hstack((noise_p,noise_gp.basis(z)))

  # add the prior covariance and basis 
  full_sigma = noise_sigma + prior_gp.covariance(z,z)
  full_p = np.hstack((noise_p,prior_gp.basis(z)))
  full_mu = np.zeros(len(z))  

  # returns the indices of outliers 
  outliers,fit = _is_outlier(d.ravel(),sd.ravel(),full_sigma,full_mu,full_p,tol)
  # dereference full_* since we will not be using them anymore
  del full_sigma,full_p,full_mu
  # best fit combination of signal and noise to the observations
  fit = fit.reshape((Nt,Nx))
  # record removed data
  d.ravel()[outliers] = np.nan
  sd.ravel()[outliers] = np.inf
  removed.ravel()[outliers] = True
  
  # toss out stations that are outliers or have infinite uncertainty
  toss = np.isinf(sd.ravel()) 
  z,d,sd = z[~toss],d.ravel()[~toss],sd.ravel()[~toss]   
  noise_p = noise_p[~toss]
  noise_sigma = noise_sigma[np.ix_(~toss,~toss)]
  # add formal data uncertainties to the noise covariance matrix
  noise_sigma += np.diag(sd**2)
  
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

  out  = (removed,fit,dx,sdx,dy,sdy)
  return out
