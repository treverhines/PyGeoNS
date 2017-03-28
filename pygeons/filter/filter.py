''' 
Defines the main filtering functions which are called by the 
executables.
'''
from __future__ import division
import numpy as np
import logging
import rbf
from pygeons.filter.gpr import gpr
from pygeons.mjd import mjd_inv,mjd
from pygeons.basemap import make_basemap
from pygeons.breaks import make_time_vert_smp, make_space_vert_smp
logger = logging.getLogger(__name__)


def pygeons_tgpr(data,sigma,cls,order=1,diff=(0,),fogm=(0.5,0.2),
                 no_annual=True,no_semiannual=True,
                 outlier_tol=4.0,procs=0,return_sample=False,
                 start_date=None,stop_date=None):
  ''' 
  Temporal Gaussian process regression. This is used to temporally
  smooth or differentiate displacements.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  sigma : float
    Prior hyperparameter describing the standard deviation of
    displacements (mm)
  
  cls : float
    Prior hyperparameter describing the characteristic time-scale (yr)
  
  order : int, optional
    Order of the polynomial basis functions.
  
  diff : int, optional
    Derivative order.
  
  fogm : 2-tuple, optional
    Hyperparameters for the FOGM noise model. The first parameter is
    the standard deviation of the white noise driving the process
    (mm/yr^0.5), and the second parameter is the cutoff frequency
    (1/yr).
    
  no_annual : bool, optional  
    Indicates whether to include annual sinusoids in the noise model.

  no_semiannual : bool, optional  
    Indicates whether to include semiannual sinusoids in the noise
    model.

  outlier_tol : float, optional
    Tolerance for outlier detection. Smaller values make the detection 
    algorithm more sensitive. This should not be set any lower than 
    about 2.0.

  procs : int, optional
    Number of subprocesses to spawn.
  
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
  print(fogm)
  logger.info('Performing temporal Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
  # make sure that the data units are displacements
  if not ((data['space_exponent'] == 1) & 
          (data['time_exponent'] == 0)):
    raise ValueError('The input dataset must have units of displacement')
  
  # convert se_sigma from mm to m
  sigma *= 1.0/1000.0
  # convert se_cls from yr to days
  cls *= 365.25
  fogm_sigma,fogm_fc = fogm
  # convert fogm_sigma from mm/yr^0.5 to m/days^0.5
  fogm_sigma *= (1.0/1000.0)*np.sqrt(1.0/365.25)
  # convert fogm_fc from 1/yr to 1/days
  fogm_fc *= 1.0/365.25
  
  # set output times
  if start_date is None:
    start_date = mjd_inv(np.min(data['time']),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd_inv(np.max(data['time']),'%Y-%m-%d')
  
  start_time = mjd(start_date,'%Y-%m-%d')  
  stop_time = mjd(stop_date,'%Y-%m-%d')  
  time = np.arange(start_time,stop_time+1)
  for dir in ['east','north','vertical']:
    post,post_sigma = gpr(data['time'][:,None],
                          data[dir].T,
                          data[dir+'_std_dev'].T,
                          (sigma,cls),
                          x=time[:,None],
                          order=order,
                          diff=diff,
                          fogm_params=(fogm_sigma,fogm_fc),
                          annual=(not no_annual),
                          semiannual=(not no_semiannual),
                          procs=procs,
                          return_sample=return_sample,
                          tol=outlier_tol)
    out[dir] = post.T
    out[dir+'_std_dev'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out['time'] = time
  return out
  
def pygeons_sgpr(data,sigma,cls,order=1,diff=(0,0),
                 return_sample=False,positions=None,
                 procs=0,outlier_tol=4.0):
  ''' 
  Spatial Gaussian process regression. This is used to spatially
  smooth or differentiate displacements or velocities.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  sigma : float
    Prior hyperparameter describing standard deviation in mm or mm/yr.
  
  cls : float
    Prior hyperparameter describing the characteristic length-scale in
    km.
  
  order : int, optional
    Order of the polynomial null space.
  
  diff : int, optional
    Derivative order.
  
  procs : int, optional
    Number of subprocesses to spawn.

  return_sample : bool, optional
    If True, then the returned dataset will be a random sample of the 
    posterior (or prior if *do_not_condition* is False), rather than 
    its expected value and uncertainty.

  positions : (str array,float array,float array), optional
    Positions for the output data set. This is a list with three 
    elements: a string array of position IDs, a float array of 
    longitudes, and a float array of latitudes. Each array must have 
    the same length. This defaults to the positions in the input data 
    set.

  outlier_tol : float, optional
    Tolerance for outlier detection. Smaller values make the detection 
    algorithm more sensitive. This should not be set any lower than 
    about 2.0.

  '''
  logger.info('Performing spatial Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
  # make sure that the data units are displacements or velocities
  if not ( (data['space_exponent'] == 1) & 
           ( (data['time_exponent'] ==  0) | 
             (data['time_exponent'] == -1) ) ):
    raise ValueError('The input dataset must have units of displacement or velocity')
    
  # convert units of se_sigma from mm or mm/yr to m or mm/day
  sigma *= (1.0/1000.0)*365.25**data['time_exponent']
  # convert units of se_cls from km to m
  cls *= 1000.0  
  
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  xy = np.array([x,y]).T
  # set output positions
  if positions is None:
    output_id = np.array(data['id'],copy=True)
    output_lon = np.array(data['longitude'],copy=True)
    output_lat = np.array(data['latitude'],copy=True)
  else:  
    output_id,output_lon,output_lat = positions

  output_x,output_y = bm(output_lon,output_lat)
  output_xy = np.array([output_x,output_y]).T 
  for dir in ['east','north','vertical']:
    post,post_sigma = gpr(xy,
                          data[dir],
                          data[dir+'_std_dev'],
                          (sigma,cls),
                          x=output_xy,
                          order=order,
                          return_sample=return_sample,
                          diff=diff,
                          tol=outlier_tol,
                          procs=procs)
    out[dir] = post
    out[dir+'_std_dev'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  # set the new lon lat and id if positions was given
  out['longitude'] = output_lon
  out['latitude'] = output_lat
  out['id'] = output_id
  return out
  

def pygeons_tfilter(data,diff=(0,),fill='none',
                    break_dates=None,**kwargs):
  ''' 
  time smoothing
  '''
  logger.info('Performing temporal RBF-FD filtering ...')
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
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

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

