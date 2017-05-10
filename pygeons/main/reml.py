''' 
Contains a function for restricted maximum likelihood (REML)
hyperparameter estimation.
'''
import numpy as np
import logging
from scipy.optimize import fmin
from pygeons.main import gpnetwork
from pygeons.main import gpstation
from pygeons.main.gptools import (composite,
                                  station_sigma_and_p)
from rbf.gauss import (_as_sparse_or_array,
                       _as_covariance,
                       likelihood)
logger = logging.getLogger(__name__)


def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint and multiple start points'''
  def pos_func(x,*blargs):
    return func(np.exp(x),*blargs)

  xopt,fopt,_,_,_ = fmin(pos_func,np.log(x0),*args,full_output=True,**kwargs)
  xopt = np.exp(xopt)
  return xopt,fopt


def reml(t,x,d,sd,
         network_model=('se-se',),
         network_params=(5.0,0.05,50.0),
         network_fix=(),
         station_model=('p0','p1'),
         station_params=(),
         station_fix=()):
  ''' 
  Returns the Restricted Maximum Likelihood (REML) estimatates of the
  unknown hyperparameters.

  Parameters
  ----------
  t : (Nt,) array
  x : (Nx,2) array
  d : (Nt,Nx) array
  s : (Nt,Nx) array
  model : str array
  params : float array
    initial guess
  fix : int array

  Returns
  -------
  theta : (P,) float array
    optimal hyperparameters 
  units : (P,) str array
    units of the hyperparameters 
  like : float array  
    likelihoods associated the optimal hyperparameters
    
  '''
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  d = np.array(d,dtype=float)
  sd = np.array(sd,dtype=float)
  diff = np.array([0,0,0])
  network_params = np.asarray(network_params,dtype=float)
  station_params = np.asarray(station_params,dtype=float)
  network_fix = np.asarray(network_fix,dtype=int)
  station_fix = np.asarray(station_fix,dtype=int)

  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')
  # flat observation times and positions
  z = np.array([t_grid.ravel(),
                x0_grid.ravel(),
                x1_grid.ravel()]).T

  # mask indicates missing data
  mask = np.isinf(sd)
  z,d,sd = z[~mask.ravel()],d[~mask],sd[~mask]
  # number of network hyperparameters
  n = len(network_params)
  # combined network and station parameters
  params = np.hstack((network_params,station_params))
  fix = np.hstack((network_fix,station_fix+n))
  free = np.array([i for i in range(len(params)) if i not in fix],dtype=int)
  
  def objective(theta):
    logger.debug('Current hyperparameters : ' + ' '.join('%0.4e' % i for i in theta))
    test_params = np.copy(params)
    test_params[free] = theta 
    test_network_params = test_params[:n]
    test_station_params = test_params[n:]
    net_gp = composite(network_model,test_network_params,gpnetwork.CONSTRUCTORS)
    sta_gp = composite(station_model,test_station_params,gpstation.CONSTRUCTORS)
    # station process
    sta_sigma,sta_p = station_sigma_and_p(sta_gp,t,mask)
    # add data noise to the diagonals of sta_sigma. Both matrices are
    # sparse so this is efficient
    obs_sigma = _as_covariance(sd)
    sta_sigma = _as_sparse_or_array(sta_sigma + obs_sigma)
    # network process
    net_sigma = net_gp._covariance(z,z,diff,diff)
    net_p = net_gp._basis(z,diff)
    # combine station gp with the network gp
    mu = np.zeros(z.shape[0])
    sigma = _as_sparse_or_array(sta_sigma + net_sigma)
    p = np.hstack((sta_p,net_p))
    del sta_sigma,net_sigma,obs_sigma,sta_p,net_p
    try:
      out = -likelihood(d,mu,sigma,p=p)
    except np.linalg.LinAlgError as err:
      logger.warning(
        'An error was raised while computing the log '
        'likelihood:\n\n%s\n' % repr(err))
      logger.warning('Returning -INF for the log likelihood')   
      out = np.inf
      
    return out  

  opt,val = fmin_pos(objective,params[free],disp=False)
  logger.info('Optimal hyperparameters : ' + ' '.join('%0.4e' % i for i in opt))
  params[free] = opt
  out_network_params = params[:n]
  out_station_params = params[n:]
  out_likelihood = -val
  return out_network_params,out_station_params,out_likelihood
