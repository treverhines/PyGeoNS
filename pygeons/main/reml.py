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


def reml(t,x,d,sd,
         network_model=('se-se',),
         network_params=(1.0,0.05,50.0),
         network_fix=(),
         station_model=('p0',),
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
  d = np.array(d,dtype=float,copy=True)
  sd = np.array(sd,dtype=float,copy=True)
  Nt,Nx = t.shape[0],x.shape[0]

  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')
  # flat observation times and positions
  z = np.array([t_grid.ravel(),
                x0_grid.ravel(),
                x1_grid.ravel()]).T

  # If a station has insufficient data for the station basis vectors
  # then mask whatever few data points it does have 
  sta_gp = gpcomp(station_model,station_params)
  Np = sta_gp.basis(np.zeros((0,1))).shape[1]
  bad_stations_bool = np.sum(~np.isinf(sd),axis=0) < Np
  bad_stations, = np.nonzero(bad_stations_bool)
  d[:,bad_stations] = np.nan
  sd[:,bad_stations] = np.inf

  # array indicating missing data
  toss = np.isinf(sd.ravel())
  z,d,sd = z[~toss],d.ravel()[~toss],sd.ravel()[~toss]

  def objective(theta):
    test_params = np.copy(params)
    test_params[is_free] = theta 
    test_network_params = TODO
    test_station_params = TODO    
    net_gp = gpcomp(network_model,test_network_params)
    sta_gp = gpcomp(station_model,test_station_params)
    # station process
    sigma,p = _station_sigma_and_p(sta_gp,t,Nx,bad_stations)
    sigma,p = sigma[np.ix_(~toss,~toss)],p[~toss]
    # network process
    sigma += net_gp.covariance(z,z)
    p = np.hstack((p,net_gp.basis(z)))
    # data uncertainty 
    rbf.gauss._diag_add(sigma,sd**2)
    # mean of the processes
    mu = np.zeros(z.shape[0])
    return -rbf.gauss.likelihood(d,mu,sigma,p=p)


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
