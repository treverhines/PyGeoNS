''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import rbf
from pygeons.mp import parmap
from rbf.gauss import PriorGaussianProcess
import logging
logger = logging.getLogger(__name__)

def _get_trend(y,d,s,x,order,diff):
  ''' 
  returns the best fitting polynomial to observations *d*, which were 
  made at *y* and have uncertainties *s*. The polynomial has order 
  *order*, is evaluated at *x*, and then differentiated by *diff*.
  '''
  if y.shape[0] == 0:
    # lstsq is unable to handle when y.shape[0]==0. In this case, 
    # return an array of zeros with shape equal to x.shape[0]
    return np.zeros(x.shape[0])

  powers = rbf.poly.powers(order,y.shape[1])
  Gobs = rbf.poly.mvmonos(y,powers) # system matrix
  W = np.diag(1.0/s) # weight matrix
  coeff = np.linalg.lstsq(W.dot(Gobs),W.dot(d))[0]
  Gitp = rbf.poly.mvmonos(x,powers,diff)
  trend = Gitp.dot(coeff) # evaluated trend at interpolation points
  return trend


def gpr(y,d,s,coeff,x=None,basis=rbf.basis.se,order=1,tol=4.0,
        diff=None,procs=0,condition=True,return_sample=False):
  '''     
  Performs Guassian process regression on the observed data. This is a 
  convenience function which initiates a *PriorGaussianProcess*, 
  conditions it with the observations, differentiates it (if 
  specified), and then evaluates the resulting *GaussianProcess* at 
  *x*. 
  
  Parameters
  ----------
  y : (N,D) array
    Observation points.

  d : (...,N) array
    Observed data at *y*.
  
  s : (...,N) array
    Data uncertainty. *np.inf* can be used to indicate that the data 
    is missing, which will cause the corresponding value in *d* to be 
    ignored.
  
  coeff : 3-tuple
    Variance, mean, and characteristic length scale for the prior 
    Gaussian process.
  
  x : (M,D) array, optional
    Evaluation points, defaults to *y*.
  
  basis : RBF instance, optional      
    Radial basis function which describes the prior covariance 
    structure. Defaults to *rbf.basis.ga*.
    
  order : int, optional
    Order of the prior null space.

  diff : (D,), optional         
    Specifies the derivative of the returned values. 

  procs : int, optional
    Distribute the tasks among this many subprocesses. This defaults 
    to 0 (i.e. the parent process does all the work).  Each task is to 
    perform Gaussian process regression for one of the (N,) arrays in 
    *d* and *s*. So if *d* and *s* are (N,) arrays then using 
    multiple process will not provide any speed improvement.
  
  condition : bool, optional
    If False then the prior Gaussian process will not be conditioned 
    with the data and the output will just be the prior or its 
    specified derivative. If the prior contains a polynomial null 
    space (i.e. order > -1), then the monomial coefficients will be 
    set to those that best fit the data. See note 3 in the 
    GaussianProcess documentation.
    
  return_sample : bool, optional
    If True then *out_mean* is a sample of the posterior, rather than 
    its expected value. *out_sigma* will then be an array of zeros, 
    since a sample has no associated uncertainty. If *return_sample* 
    is True and *condition* is False then a sample of the prior will 
    be returned.
    
  Returns 
  ------- 
  out_mean : (...,M) array
    Mean of the posterior at *x*.

  out_sigma : (...,M) array  
    One standard deviation of the posterior at *x*.

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
    gp = PriorGaussianProcess(coeff,basis=basis,order=order,dim=x.shape[1])
    # if the uncertainty is inf then the data is considered missing
    is_missing = np.isinf(s[i])
    # start by just ignoring missing data
    ignore = np.copy(is_missing)
    if condition:
      # iteratively condition and identify outliers
      while True:
        gpi = gp.condition(y[~ignore],d[i,~ignore],sigma=s[i,~ignore])
        res = np.abs(gpi.mean(y) - d[i])/s[i]
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
      
      gp = gpi    
               
    gp = gp.differentiate(diff)
    if return_sample:
      out_mean_i = gp.draw_sample(x)
      out_sigma_i = np.zeros_like(out_mean_i)
    else:
      out_mean_i,out_sigma_i = gp.mean_and_uncertainty(x)

    if gp.order != -1:
      # Read note 3 in the GaussianProcess documentation. If a 
      # polynomial null space exists, then I am setting the monomial 
      # coefficients to be the coefficients that best fit the data. 
      # This deviates from the default behavior for a 
      # GaussianProcess, which sets the monomial coefficients to 
      # zero. Note, that there is no attempt to identify outliers 
      # when determining the trend
      trend = _get_trend(y[~ignore],d[i,~ignore],
                         s[i,~ignore],x,order,diff)
      out_mean_i += trend

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

