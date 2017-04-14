''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import logging
import rbf.gauss
from pygeons.mp import parmap
from pygeons.filter.gprocs import gpcomp
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
  
  returns a boolean array indicating which points are outliers
  '''
  out = np.zeros(d.shape[0],dtype=bool)
  while True:
    q = sum(~out)
    a = sigma[np.ix_(~out,~out)] + np.diag(s[~out]**2)
    b = rbf.gauss._cholesky_block_inv(a,p[~out])
    c = np.empty((d.shape[0],b.shape[0]))
    c[:,:q] = sigma[:,~out]
    c[:,q:] = p
    r = np.empty(b.shape[0])
    r[:q] = d[~out] - mu[~out]
    r[q:] = 0.0
    pred = mu + c.dot(b).dot(r)
    res = np.abs(pred - d)/s
    rms = np.sqrt(np.mean(res[~out]**2))
    if np.all(out == (res > tol*rms)):
      break

    else:
      out = (res > tol*rms)

  return out


def strain(t,x,d,s,
           prior_model,prior_params,
           out_t=None,
           out_x=None,
           diff=None,
           noise_model='null',
           noise_params=(),
           tol=4.0,
           procs=0,
           return_sample=False,
           comm=False,
           offsets=True):
  ''' 
  Computes deformation gradients from displacement data.
  
  Parameters
  ----------
  t : (Nt,1) array
    Observation times.
  
  x : (Nx,2) array
    Observation positions.  

  d : (Nt,Nx) array
    Grid of observations at time *t* and position *x*. 
  
  s : (Nt,Nx) array
    Grid of observation uncertainties
  
  prior_model : str
    String specifying the prior model
  
  prior_params : 2-tuple
    Hyperparameters for the prior model.
  
  out_t : (Mt,) array, optional
    Output times
  
  out_x : (Mx,2) array, optional
    Output positions  

  diff : (3,), optional         
    Tuple specifying the derivative of the returned values. First
    element is the time derivative, and the second two elements are
    the space derivatives.

  noise_model : str, optional
    String specifying the noise model
    
  noise_params : 2-tuple, optional
    Hyperparameters for the noise model
    
  tol : float, optional
    Tolerance for the outlier detection algorithm.
    
  procs : int, optional
    Distribute the tasks among this many subprocesses. 
  
  return_sample : bool, optional
    If True then *out_mean* is a sample of the posterior.

  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.asarray(d,dtype=float)
  s = np.asarray(s,dtype=float)
  if diff is None:
    diff = np.zeros(3,dtype=int)

  if out_t is None:
    out_t = t

  if out_x is None:
    out_x = x

  d = d.flatten()
  s = s.flatten()
  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')  
  # flat observation times and positions
  z = np.array([t_grid.flatten(),
                x0_grid.flatten(),
                x1_grid.flatten()]).T

  t_grid,x0_grid = np.meshgrid(out_t,out_x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(out_t,out_x[:,1],indexing='ij')  
  # flat observation times and positions
  out_z = np.array([t_grid.flatten(),
                    x0_grid.flatten(),
                    x1_grid.flatten()]).T

  # create basis functions for common modes and station offsets
  p = np.zeros((t.shape[0]*x.shape[0],0))
  if offsets:
    # create station offset basis functions
    offsets_p = np.zeros((t.shape[0],x.shape[0],x.shape[0]))
    for i in range(x.shape[0]): offsets_p[:,i,i] = 1.0
    offsets_p = offsets_p.reshape((t.shape[0]*x.shape[0],x.shape[0])).T
    p = np.hstack((p,offsets_p))

  if comm:
    # create common mode error basis functions
    comm_p = np.zeros((t.shape[0],x.shape[0],t.shape[0]))
    for i in range(t.shape[0]): comm_p[i,:,i] = 1.0
    comm_p = comm_p.reshape((t.shape[0]*x.shape[0],t.shape[0])).T
    p = np.hstack((p,comm_p))
  
  # Do a data check to make sure that there are no stations or times
  # with no data? is it needed

  # if the uncertainty is inf then the data is considered missing
  # and will be tossed out
  toss = np.isinf(s)
  z = z[~toss] 
  d = d[~toss]
  s = s[~toss]
  p = p[~toss]

  prior_gp = gpcomp(prior_model,prior_params)
  noise_gp = gpcomp(noise_model,noise_params)    
  full_gp  = prior_gp + noise_gp 

  full_sigma = full_gp.covariance(z,z) # model covariance
  full_mu = full_gp.mean(z) # model mean
  full_p  = np.hstack((full_gp.basis(z),p))
  toss = _is_outlier(d,s,full_sigma,full_mu,full_p,tol)
  logger.info('Detected %s outliers out of %s observations' % (sum(toss),len(toss)))
  z = z[~toss] 
  d = d[~toss]
  s = s[~toss]
  p = p[~toss]

  noise_sigma = np.diag(s**2) + noise_gp.covariance(z,z)
  noise_p     = np.hstack((noise_gp.basis(z),p))
  # condition the prior with the data
  post_gp = prior_gp.condition(z,d,sigma=noise_sigma,p=noise_p)
  post_gp = post_gp.differentiate(diff)
  if return_sample:
    out_mean = post_gp.sample(out_z)
    out_sigma = np.zeros_like(out_mean_i)
  else:
    out_mean,out_sigma = post_gp.meansd(out_z)

  out_mean = out_mean.reshape((out_t.shape[0],out_x.shape[0]))
  out_sigma = out_sigma.reshape((out_t.shape[0],out_x.shape[0]))
  return out_mean,out_sigma
