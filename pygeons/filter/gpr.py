''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import logging
from pygeons.mp import parmap
from pygeons.filter.gprocs import gpcomp
logger = logging.getLogger(__name__)


def gpr(y,d,s,
        prior_model,prior_params,
        x=None,
        diff=None,
        noise_model='null',
        noise_params=(),
        tol=4.0,
        procs=0,
        return_sample=False):
  ''' 
  Performs Guassian process regression. 
  
  Parameters
  ----------
  y : (N,D) array
    Observation points.

  d : (...,N) array
    Observed data at *y*.
  
  s : (...,N) array
    Data uncertainty.
  
  prior_model : str
    String specifying the prior model
  
  prior_params : 2-tuple
    Hyperparameters for the prior model.
  
  x : (M,D) array, optional
    Evaluation points.

  diff : (D,), optional         
    Specifies the derivative of the returned values. 

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
  y = np.asarray(y,dtype=float)
  d = np.asarray(d,dtype=float)
  s = np.asarray(s,dtype=float)
  if diff is None:
    diff = np.zeros(y.shape[1],dtype=int)

  if x is None:
    x = y

  m = x.shape[0]
  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  s = s.reshape((q,n))

  def task(i):
    logger.debug('Processing dataset %s of %s ...' % (i+1,q))
    if np.any(s[i] <= 0.0):
      raise ValueError(
        'At least one datum has zero or negative uncertainty.')
    
    # if the uncertainty is inf then the data is considered missing
    is_missing = np.isinf(s[i])
    # start by just ignoring missing data
    ignore = np.copy(is_missing)
    # iteratively condition and identify outliers
    while True:
      prior_gp = gpcomp(prior_model,prior_params)
      noise_gp = gpcomp(noise_model,noise_params)    
      full_gp  = prior_gp + noise_gp 
      yi,di,si = y[~ignore],d[i,~ignore],s[i,~ignore]
      fit_gp   = full_gp.condition(yi,di,sigma=si)
      # compute residuals using all the data
      res = np.abs(fit_gp.mean(y) - d[i])/s[i]
      # give missing data infinite residuals
      res[is_missing] = np.inf
      rms = np.sqrt(np.mean(res[~ignore]**2))
      if np.all(ignore == (res > tol*rms)):
        # compile data satistics
        missing_count = np.sum(is_missing)
        outlier_count = np.sum(ignore) - missing_count
        data_count = len(ignore) - missing_count
        logger.debug('observations: %s, detected outliers: %s' % (data_count,outlier_count))
        break
      else:  
        ignore = (res > tol*rms)
    
    sigma  = np.diag(si**2) + noise_gp.covariance(yi,yi)
    p      = noise_gp.basis(yi)
    post_gp = prior_gp.condition(yi,di,sigma=sigma,p=p)
    post_gp = post_gp.differentiate(diff)
    if return_sample:
      out_mean_i = post_gp.sample(x)
      out_sigma_i = np.zeros_like(out_mean_i)
    else:
      out_mean_i,out_sigma_i = post_gp.meansd(x)

    return out_mean_i,out_sigma_i
    
  def task_with_error_catch(i):    
    try:
      return task(i)

    except Exception as err:  
      logger.info(
        'An error was raised when processing dataset %s, "%s", ' 
        'The returned expected values and uncertainties will be NaN '
        'and INF, respectively.' % ((i+1),err))
        
      out_mean_i = np.full(x.shape[0],np.nan)
      out_sigma_i = np.full(x.shape[0],np.inf)
      return out_mean_i,out_sigma_i

  out = parmap(task_with_error_catch,range(q),workers=procs)
  out_mean = np.array([k[0] for k in out])
  out_sigma = np.array([k[1] for k in out])
  out_mean = out_mean.reshape(bcast_shape + (m,))
  out_sigma = out_sigma.reshape(bcast_shape + (m,))
  return out_mean,out_sigma
