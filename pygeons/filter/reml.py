# -*- coding: utf-8 -*-
''' 
Contains a function for restricted maximum likelihood (REML)
hyperparameter estimation.
'''
import numpy as np
import logging
from scipy.optimize import fmin
from pygeons.mp import parmap
from pygeons.filter.gpr import gpseasonal,gpfogm,gppoly,gpse                       
logger = logging.getLogger(__name__)


def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint and multiple start points'''
  def pos_func(x,*blargs):
    return func(np.exp(x),*blargs)

  xopt,fopt,_,_,_ = fmin(pos_func,np.log(x0),*args,full_output=True,**kwargs)
  xopt = np.exp(xopt)
  return xopt,fopt


def reml(y,d,s,model,params,
         fix=(), 
         order=1,
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
  
  model : str, optional
    string indicating the stochastic model being optimized. Can either
    be 'se','fogm', or 'se+fogm'. The required length of *params* will
    change depending on the number of hyperparameters in the model.
    'se' and 'fogm' take 2 hyperparameters, and 'se+fogm' takes 4
    hyperparameters.
    
  params : (P,) array 
    Initial guess for the hyperparameters. If 4 are specified then the
    model consists of SE and FOGM. If 2 are specified then the model
    consists of SE
  
  fix : (L,) int array
    Indices of hyperparameters that will remain fixed
  
  order : int, optional
    Order of the polynomial improper basis functions.
  
  annual : bool, optional  
    Indicates whether to include annual sinusoids in the model.

  semiannual : bool, optional  
    Indicates whether to include semiannual sinusoids in the model.

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
    
  d : (P,) str array
    units of the hyperparameters 
    
  '''
  y = np.asarray(y,dtype=float)
  d = np.asarray(d,dtype=float)
  s = np.asarray(s,dtype=float)
  params = np.asarray(params,dtype=float)
  fix = np.asarray(fix,dtype=int)

  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  s = s.reshape((q,n))

  if model == 'se':
    units = np.array(['a[{0}]','b[{1}]'])
    if len(params) != 2:
      raise ValueError(
        'exactly 2 parameters must be specified for the *se* '
        'covariance function')

  elif model == 'fogm': 
    units = np.array(['c[{0}*{1}^-0.5]','d[{1}^-1]'])
    if len(params) != 2:
      raise ValueError(
        'exactly 2 parameters must be specified for the *fogm* '
        'covariance function')

  elif (model == 'se+fogm'):
    units = np.array(['a[{0}]','b[{1}]','c[{0}*{1}^-0.5]','d[{1}^-1]'])
    if len(params) != 4:
      raise ValueError(
        'exactly 4 parameters must be specified for the *se+fogm* '
        'covariance function')
        
  else:
    raise ValueError('*%s* is not a valid covariance function' % model)

  is_free = np.ones(params.shape,dtype=bool)
  is_free[fix] = False

  def objective(theta,pos,data,sigma):
    test_params = np.copy(params)
    test_params[is_free] = theta 
    if model == 'se':
      # cov(t,t') = a^2 exp(-|t - t'|^2/b^2)
      gp = gpse((0.0,test_params[0]**2,test_params[1])) 
      
    elif model == 'fogm':
      #cov(t,t') = c^2/(4 pi d) exp(-2 pi d |t - t'|)
      gp = gpfogm(test_params[0],test_params[1])

    elif model == 'se+fogm':
      # cov(t,t') = a^2 exp(-|t - t'|^2/b^2) + c^2/(4 pi d) exp(-2 pi d |t - t'|)
      gp  = gpse((0.0,test_params[0]**2,test_params[1])) 
      gp += gpfogm(test_params[2],test_params[3])
    
    gp += gppoly(order)
    if annual | semiannual:
      gp += gpseasonal(annual,semiannual)
    
    # return negative log likelihood
    return -gp.likelihood(pos,data,sigma)  

  def task(i):
    logger.debug('Finding REML hyperparameters for dataset %s of %s ...' % (i+1,q))
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
