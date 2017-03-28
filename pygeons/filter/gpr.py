''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import rbf
import logging
from scipy.optimize import fmin
from pygeons.mp import parmap
from rbf.gauss import (gpbfci,gppoly,gpse,gpexp,GaussianProcess,
                       _zero_mean,_zero_covariance,_empty_basis)
logger = logging.getLogger(__name__)


def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint and multiple start points'''
  def pos_func(x,*blargs):
    return func(np.exp(x),*blargs)

  xopt,fopt,_,_,_ = fmin(pos_func,np.log(x0),*args,full_output=True,**kwargs)
  xopt = np.exp(xopt)
  return xopt,fopt


def gpnull():
  return GaussianProcess(_zero_mean,_zero_covariance,basis=_empty_basis)


def gpseasonal(annual,semiannual):
  ''' 
  Returns a *GaussianProcess* with annual and semiannual terms as
  improper basis functions.
  '''
  def basis(x):
    out = np.zeros((x.shape[0],0))
    if annual:
      # note that x is in days
      terms = np.array([np.sin(2*np.pi*x[:,0]/365.25),
                        np.cos(2*np.pi*x[:,0]/365.25)]).T
      out = np.hstack((out,terms))
      
    if semiannual:
      terms = np.array([np.sin(4*np.pi*x[:,0]/365.25),
                        np.cos(4*np.pi*x[:,0]/365.25)]).T
      out = np.hstack((out,terms))
    
    return out
    
  return gpbfci(basis,dim=1)


def gpfogm(s,fc):
  ''' 
  Returns a *GaussianProcess* describing an first-order Gauss-Markov
  process. The autocovariance function is
    
     K(t) = s^2/(4*pi*fc) * exp(-2*pi*fc*|t|)  
   
  which has the corresponding power spectrum 
  
     P(f) = s^2/(4*pi^2 * (f^2 + fc^2))
  
  *fc* can be interpretted as a cutoff frequency which marks the
  transition to a flat power spectrum and a power spectrum that decays
  with a spectral index of two. Thus, when *fc* is close to zero, the
  power spectrum resembles that of Brownian motion.
  '''
  coeff = s**2/(4*np.pi*fc)
  cls   = 1.0/(2*np.pi*fc)
  return gpexp((0.0,coeff,cls))

     
def gpr(y,d,s,se_params,x=None,
        order=1,
        diff=None,
        fogm_params=None,
        annual=False,
        semiannual=False,
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
  
  x : (M,D) array, optional
    Evaluation points.

  se_params : 2-tuple
    Hyperparameters for the squared-exponential prior model.
  
  order : int, optional
    Order of the polynomial improper basis functions.

  diff : (D,), optional         
    Specifies the derivative of the returned values. 

  fogm_params : 2-tuple, optional
    Hyperparameters for the first-order Gauss-Markov (FOGM) noise
    model. 
    
  annual : bool, optional  
    Indicates whether to include annual sinusoids in the noise model.

  semiannual : bool, optional  
    Indicates whether to include semiannual sinusoids in the noise
    model.
    
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
    prior_gp  = gpse((0.0,se_params[0]**2,se_params[1]),dim=x.shape[1]) 
    prior_gp += gppoly(order)
    noise_gp  = gpnull()
    if annual | semiannual:
      noise_gp += gpseasonal(annual,semiannual)

    if fogm_params is not None:
      noise_gp += gpfogm(*fogm_params)
      
    # if the uncertainty is inf then the data is considered missing
    is_missing = np.isinf(s[i])
    # start by just ignoring missing data
    ignore = np.copy(is_missing)
    # iteratively condition and identify outliers
    while True:
      yi,di,si = y[~ignore],d[i,~ignore],s[i,~ignore]
      # create noise covariance matrix
      sigma = np.diag(si**2) + noise_gp.covariance(yi,yi)
      # create improper noise basis vectors
      p = noise_gp.basis(yi)
      # condition the prior
      post_gp = prior_gp.condition(yi,di,sigma=sigma,p=p)
      # compute residuals using all the data
      res = np.abs(post_gp.mean(y) - d[i])/s[i]
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

    except np.linalg.LinAlgError:  
      logger.info(
        'Could not process dataset %s. This may be due to '
        'insufficient data. The returned expected values and '
        'uncertainties will be NaN and INF, respectively.' % (i+1))
      out_mean_i = np.full(x.shape[0],np.nan)
      out_sigma_i = np.full(x.shape[0],np.inf)
      return out_mean_i,out_sigma_i

  out = parmap(task_with_error_catch,range(q),workers=procs)
  out_mean = np.array([k[0] for k in out])
  out_sigma = np.array([k[1] for k in out])
  out_mean = out_mean.reshape(bcast_shape + (m,))
  out_sigma = out_sigma.reshape(bcast_shape + (m,))
  return out_mean,out_sigma


def reml(y,d,s,params,
         fix=None, 
         order=1,
         restarts=1,
         annual=False,
         semiannual=False,
         procs=0):
  ''' 
  Returns the Restricted Maximum Likelihood (REML) estimatates of the
  unknown hyperparameters.

  Parameters
  ----------
  y : (N,D) array
    Observation points.

  d : (...,N) array
    Observed data at *y*.
  
  s : (...,N) array
    Data uncertainty.
  
  params : (4,) array
    Initial guess for the hyperparameters.
  
  fix : (P,) int array
    Indices of the parameters which will be fixed at the initial guess
  
  order : int, optional
    Order of the polynomial improper basis functions.
  
  restarts : int, optional
    Number of restarts
    
  guess_range : float, optional
    Range of values used for the initial guesses in log space.

  annual : bool, optional  
    Indicates whether to include annual sinusoids in the noise model.

  semiannual : bool, optional  
    Indicates whether to include semiannual sinusoids in the noise
    model.

  procs : int, optional
    Distribute the tasks among this many subprocesses. 

  Returns
  -------
  out : (...,4) array
    hyperparameters for each dataset. The hyperparameters are the
    standard deviation for the SE, the characteristic-length scale for
    the SE, the standard deviation for the FOGM, and the cutoff
    frequency for the FOGM.
    
  '''
  params = np.asarray(params,dtype=float)
  y = np.asarray(y,dtype=float)
  d = np.asarray(d,dtype=float)
  s = np.asarray(s,dtype=float)
  if fix is None:
    fix = np.zeros(0,dtype=int)

  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  s = s.reshape((q,n))

  # index of parameters that will be estimated
  is_free = np.ones(len(params),dtype=bool)
  is_free[fix] = False
  dof = np.sum(is_free)

  def objective(theta,pos,data,sigma):
    test_params = np.copy(params)
    test_params[is_free] = theta 
    model  = gpse((0.0,test_params[0]**2,test_params[1])) 
    model += gpfogm(test_params[2],test_params[3])
    model += gppoly(order)
    if annual | semiannual:
      model += gpseasonal(annual,semiannual)
    
    # return negative log likelihood
    return -model.likelihood(pos,data,sigma)  

  def task(i):
    logger.debug('Finding REML hyperparameters for dataset %s of %s ...' % (i+1,q))
    # if the uncertainty is inf then the data is considered missing
    is_missing = np.isinf(s[i])
    yi,di,si = y[~is_missing],d[i,~is_missing],s[i,~is_missing]
    opt = fmin_pos(objective,params[is_free],args=(yi,di,si))
    out_params = np.copy(params)
    out_params[is_free] = opt
    return out_params

  def task_with_error_catch(i):    
    try:
      return task(i)

    except np.linalg.LinAlgError:  
      logger.info(
        'Could not process dataset %s. This may be due to '
        'insufficient data. The returned hyperparameters will '
        'be NaN.' % (i+1))
      return np.full(len(params),np.nan)
  
  out = parmap(task,range(q),workers=procs)
  out = np.array(out)
  out = out.reshape(bcast_shape + (len(params),))
  return out
