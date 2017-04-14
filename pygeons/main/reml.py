# -*- coding: utf-8 -*-
''' 
Contains a function for restricted maximum likelihood (REML)
hyperparameter estimation.
'''
import numpy as np
import logging
from scipy.optimize import fmin
from pygeons.mp import parmap
from pygeons.filter.gprocs import gpcomp,get_units
logger = logging.getLogger(__name__)


def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint and multiple start points'''
  def pos_func(x,*blargs):
    return func(np.exp(x),*blargs)

  xopt,fopt,_,_,_ = fmin(pos_func,np.log(x0),*args,full_output=True,**kwargs)
  xopt = np.exp(xopt)
  return xopt,fopt


def reml(y,d,s,model,params,
         fix=(),procs=0):
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
  
  model : str,
    string indicating the stochastic model being optimized.
    
  params : (P,) array 
    Initial guess for the hyperparameters.
  
  fix : (L,) int array
    Indices of hyperparameters that will remain fixed
  
  procs : int, optional
    Distribute the tasks among this many subprocesses. 
  
  Returns
  -------
  a : (...,P) float array
    hyperparameters for each dataset. The hyperparameters are the
    standard deviation for the SE, the characteristic-length scale for
    the SE, the standard deviation for the FOGM, and the cutoff
    frequency for the FOGM.
  
  b : (...) float array  
    likelihoods associated with each set of hyperparameters

  c : (...) int array
    number of observations used to constrain the hyperparameters
    
  units : (P,) str array
    units of the hyperparameters 
    
  '''
  y = np.asarray(y,dtype=float)
  d = np.asarray(d,dtype=float)
  s = np.asarray(s,dtype=float)
  params = np.asarray(params,dtype=float)
  fix = np.asarray(fix,dtype=int)
  is_free = np.ones(params.shape,dtype=bool)
  is_free[fix] = False

  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  s = s.reshape((q,n))
  units = get_units(model)

  def objective(theta,pos,data,sigma):
    test_params = np.copy(params)
    test_params[is_free] = theta 
    gp = gpcomp(model,test_params)
    # return negative log likelihood
    try:
      return -gp.likelihood(pos,data,sigma)  
    except Exception as err:
      logger.info(
        'An error was raised while evaluating the likelihood for '
        'parameters %s, "%s". The returned likelihood will be INF' 
        % (test_params,err))
      return np.inf

  def task(i):
    logger.debug('Finding REML hyperparameters for dataset %s of %s ...' % (i+1,q))
    if np.any(s[i] <= 0.0):
      logger.info(
        'At least one datum has zero or negative uncertainty. The '
        'returned hyperparameters will be NaN.')
      return np.full_like(params,np.nan),np.nan,0

    # if the uncertainty is inf then the data is considered missing
    is_missing = np.isinf(s[i])
    yi,di,si = y[~is_missing],d[i,~is_missing],s[i,~is_missing]
    opt,l = fmin_pos(objective,params[is_free],args=(yi,di,si),disp=False)
    out_params = np.copy(params)
    out_params[is_free] = opt
    return out_params,l,np.sum(~is_missing)

  out = parmap(task,range(q),workers=procs)
  # parameter estimates   
  a = np.array([i[0] for i in out])
  a = a.reshape(bcast_shape + params.shape)
  # associated likelihoods
  b = np.array([i[1] for i in out])
  b = b.reshape(bcast_shape)
  # number of observations 
  c = np.array([i[2] for i in out])
  c = c.reshape(bcast_shape)
  return a,b,c,units
