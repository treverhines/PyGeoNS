''' 
Defines the main filtering functions which are called by the 
executables.
'''
from __future__ import division
import numpy as np
import rbf.filter
import rbf.gpr
import logging
from pygeons.mjd import mjd_inv,mjd
from pygeons.datacheck import check_data
from pygeons.basemap import make_basemap
from pygeons.breaks import make_time_vert_smp, make_space_vert_smp
logger = logging.getLogger(__name__)

def pygeons_tgpr(data,sigma,cls,order=1,diff=(0,),
                 do_not_condition=False,return_sample=False,
                 start_date=None,stop_date=None,procs=0):
  ''' 
  Temporal Gaussian process regression
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  sigma : float
    Prior standard deviation. Units of millimeters**p years**q, where 
    p and q are *time_exponent* and *space_exponent* from the data 
    dictionary.
  
  cls : float
    Characteristic length-scale in years.
  
  order : int, optional
    Order of the polynomial null space.
  
  diff : int, optional
    Derivative order.
  
  procs : int, optional
    Number of subprocesses to spawn.
  
  do_not_condition : bool, optional
    If True, then the prior Gaussian process will not be conditioned 
    with the data and the returned dataset will just be the prior or 
    its specified derivative.

  return_sample : bool, optional
    If True, then the returned dataset will be a random sample of the 
    posterior (or prior if *do_not_condition* is False), rather than 
    its expected value and uncertainty.
    
  start_date : str, optional
    Start date for the output data set, defaults to the start date for 
    the input data set.
    
  stop_date : str, optional
    Stop date for the output data set, defaults to the stop date for 
    the input data set.
    
  '''
  logger.info('Performing temporal Gaussian process regression ...')
  check_data(data)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
  
  # convert units of sigma from mm**p years**q to m**p days**q
  sigma *= 0.001**data['space_exponent']*365.25**data['time_exponent']
  # convert units of cls from years to days
  cls *= 365.25
  # set output times
  if start_date is None:
    start_date = mjd_inv(np.min(data['time']),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd_inv(np.max(data['time']),'%Y-%m-%d')
  
  start_time = mjd(start_date,'%Y-%m-%d')  
  stop_time = mjd(stop_date,'%Y-%m-%d')  
  time = np.arange(start_time,stop_time+1)
  # scaling factor for numerical stability
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.gpr.gpr(
                        data['time'][:,None],data[dir].T,
                        data[dir+'_std_dev'].T,(0.0,sigma**2,cls),
                        x=time[:,None],basis=rbf.basis.ga,
                        order=order,condition=(not do_not_condition),
                        return_sample=return_sample,diff=diff,
                        procs=procs)
    out[dir] = post.T
    out[dir+'_std_dev'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out['time'] = time
  return out
  

def pygeons_sgpr(data,sigma,cls,order=1,diff=(0,0),
                 do_not_condition=False,return_sample=False,
                 positions=None,procs=0):
  ''' 
  Temporal Gaussian process regression
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  sigma : float
    Prior standard deviation. Units of millimeters**p years**q, where 
    p and q are *time_exponent* and *space_exponent* from the data 
    dictionary.
  
  cls : float
    Characteristic length-scale in kilometers.
  
  order : int, optional
    Order of the polynomial null space.
  
  diff : int, optional
    Derivative order.
  
  procs : int, optional
    Number of subprocesses to spawn.

  do_not_condition : bool, optional
    If True, then the prior Gaussian process will not be conditioned 
    with the data and the returned dataset will just be the prior or 
    its specified derivative.

  return_sample : bool, optional
    If True, then the returned dataset will be a random sample of the 
    posterior (or prior if *do_not_condition* is False), rather than 
    its expected value and uncertainty.

  positions : (N,2) array, optional
    Positions for the output data set, defaults to positions in the 
    input data set. 
    
  '''
  logger.info('Performing spatial Gaussian process regression ...')
  check_data(data)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  # convert units of sigma from mm**p years**q to m**p days**q
  sigma *= 0.001**data['space_exponent']*365.25**data['time_exponent']
  # convert units of cls from km to m
  cls *= 1000.0  
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  xy = np.array([x,y]).T
  # set output positions
  if positions is None:
    positions = np.array([data['longitude'],data['latitude']],copy=True).T
    ids = np.array(data['id'],copy=True)
  else:  
    positions = np.asarray(positions)
    ids = np.array(['%04d' % i for i in range(positions.shape[0])])

  output_x,output_y = bm(positions[:,0],positions[:,1])
  output_xy = np.array([output_x,output_y]).T 
  # scaling factor for numerical stability
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.gpr.gpr(
                        xy,data[dir],data[dir+'_std_dev'],
                        (0.0,sigma**2,cls),
                        x=output_xy,basis=rbf.basis.ga,
                        order=order,condition=(not do_not_condition),
                        return_sample=return_sample,diff=diff,
                        procs=procs)
    out[dir] = post
    out[dir+'_std_dev'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  # set the new lon lat and id if positions was given
  out['longitude'] = positions[:,0]
  out['latitude'] = positions[:,1]
  out['id'] = ids
  return out
  

def pygeons_tfilter(data,diff=(0,),fill='none',
                    break_dates=None,**kwargs):
  ''' 
  time smoothing
  '''
  logger.info('Performing temporal RBF-FD filtering ...')
  check_data(data)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())  

  vert,smp = make_time_vert_smp(break_dates)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        data['time'][:,None],data[dir].T,
                        sigma=data[dir+'_std_dev'].T,
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,
                        **kwargs)
    out[dir] = post.T
    out[dir+'_std_dev'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  return out


def pygeons_sfilter(data,diff=(0,0),fill='none',
                    break_lons=None,break_lats=None,
                    break_conn=None,**kwargs):
  ''' 
  space smoothing
  '''
  logger.info('Performing spatial RBF-FD filtering ...')
  check_data(data)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  data.check_self_consistency()
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  vert,smp = make_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        pos,data[dir],
                        sigma=data[dir+'_std_dev'],
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,     
                        **kwargs)
    out[dir] = post
    out[dir+'_std_dev'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  return out

