''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import rbf
from pygeons.mp import parmap
from rbf.gpr import PriorGaussianProcess
import logging
logger = logging.getLogger(__name__)

def _get_trend(y,d,sigma,x,order,diff):
  ''' 
  returns the best fitting polynomial to observations *d*, which were 
  made at *y* and have uncertainties *sigma*. The polynomial has order 
  *order*, is evaluated at *x*, and then differentiated by *diff*.
  '''
  if y.shape[0] == 0:
    # lstsq is unable to handle when y.shape[0]==0. In this case, 
    # return an array of zeros with shape equal to x.shape[0]
    return np.zeros(x.shape[0])

  powers = rbf.poly.powers(order,y.shape[1])
  Gobs = rbf.poly.mvmonos(y,powers) # system matrix
  W = np.diag(1.0/sigma) # weight matrix
  coeff = np.linalg.lstsq(W.dot(Gobs),W.dot(d))[0]
  Gitp = rbf.poly.mvmonos(x,powers,diff)
  trend = Gitp.dot(coeff) # evaluated trend at interpolation points
  return trend


def gpr(y,d,sigma,coeff,x=None,basis=rbf.basis.se,order=1,
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
  
  sigma : (...,N) array
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
    *d* and *sigma*. So if *d* and *sigma* are (N,) arrays then using 
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
  sigma = np.asarray(sigma,dtype=float)
  if diff is None:
    diff = np.zeros(y.shape[1],dtype=int)

  if x is None:
    x = y

  m = x.shape[0]
  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  sigma = sigma.reshape((q,n))

  def doit(i):
    logger.debug('Performing GPR on data set %s of %s ...' % (i+1,q))
    gp = PriorGaussianProcess(coeff,basis=basis,order=order,dim=x.shape[1])
    # ignore data that has infinite uncertainty
    is_finite = ~np.isinf(sigma[i])
    if condition:
      gp = gp.recursive_condition(y[is_finite],d[i,is_finite],
                                  sigma=sigma[i,is_finite])

    gp = gp.differentiate(diff)
    try:
      if return_sample:
        out_mean_i = gp.draw_sample(x)
        out_sigma_i = np.zeros_like(out_mean_i)
      else:
        out_mean_i,out_sigma_i = gp.mean_and_uncertainty(x)

      if gp.order != -1:
        # Read note 3 in the GaussianProcess documentation. If a 
        # polynomial null space exists, then I am setting the monomial 
        # coefficients to be the coefficients that best fit the data. 
        # This deviates from the default behavior for a GaussianProcess, 
        # which sets the monomial coefficients to zero.
        trend = _get_trend(y[is_finite],d[i,is_finite],
                           sigma[i,is_finite],x,order,diff)
        out_mean_i += trend

    except np.linalg.LinAlgError:
      logger.info(
        'Could not compute the expected values and uncertainties for '
        'the Gaussian process. This may be due to insufficient data. '
        'The returned expected values and uncertainties will be NaN '
        'and INF, respectively.')
      out_mean_i = np.empty(x.shape[0])
      out_mean_i[:] = np.nan
      out_sigma_i = np.empty(x.shape[0])
      out_sigma_i[:] = np.inf

    return out_mean_i,out_sigma_i

  out = parmap(doit,range(q),workers=procs)
  out_mean = np.array([k[0] for k in out])
  out_sigma = np.array([k[1] for k in out])
  out_mean = out_mean.reshape(bcast_shape + (m,))
  out_sigma = out_sigma.reshape(bcast_shape + (m,))
  return out_mean,out_sigma

