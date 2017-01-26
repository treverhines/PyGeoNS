''' 
Defines functions which are called by the PyGeoNS executables. These 
functions are thin wrappers which handle the tedious tasks like 
conversions from geodetic to cartesian coordinate systems or between 
dates and decimal years. The wrappers are designed to have input 
arguments that can be easily specified through the command line. This 
limits the full functionality of the functions being called in the 
interest of usability.

Each of these function take a data dictionary as input. The data 
dictionary contains the following items:

  time : (Nt,) array      # observation time in days since 1970-01-01
  longitude : (Nx,) array
  latitude : (Nx,) array
  id : (Nx,) array        # station ID
  east : (Nt,Nx) array
  north : (Nt,Nx) array
  vertical : (Nt,Nx) array
  east_std : (Nt,Nx) array
  north_std : (Nt,Nx) array
  vertical_std : (Nt,Nx) array
  time_exponent : int
  space_exponent : int

'''
from __future__ import division
import numpy as np
import rbf.filter
import rbf.gpr
import logging
from rbf.filter import _get_mask
from pygeons.mjd import mjd_inv,mjd
from pygeons.datadict import DataDict
from pygeons.basemap import make_basemap
from pygeons.breaks import make_time_vert_smp, make_space_vert_smp
logger = logging.getLogger(__name__)


def pygeons_tgpr(data,sigma,cls,order=1,diff=(0,),
                 procs=0,fill='none',
                 output_start_date=None,
                 output_stop_date=None):
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
  
  output_start_date : str, optional
    Start date for the output data set, defaults to the start date for 
    the input data set.
    
  output_stop_date : str, optional
    Stop date for the output data set, defaults to the stop date for 
    the input data set.
    
  '''
  data.check_self_consistency()
  out = DataDict(data)
  
  # convert units of sigma from mm**p years**q to m**p days**q
  sigma *= 0.001**data['space_exponent']*365.25**data['time_exponent']
  # convert units of cls from years to days
  cls *= 365.25
  # set output times
  if output_start_date is None:
    output_start_date = mjd_inv(np.min(data['time']),'%Y-%m-%d')

  if output_stop_date is None:
    output_stop_date = mjd_inv(np.max(data['time']),'%Y-%m-%d')
  
  output_start_time = mjd(output_start_date,'%Y-%m-%d')  
  output_stop_time = mjd(output_stop_date,'%Y-%m-%d')  
  output_times = np.arange(output_start_time,output_stop_time+1)
  # scaling factor for numerical stability
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.gpr.gpr(
                        data['time'][:,None],data[dir].T,data[dir+'_std'].T,(0.0,sigma**2,cls),
                        x=output_times[:,None],basis=rbf.basis.ga,order=order,diff=diff,procs=procs)
    # loop over the stations and mask the time series based on the 
    # argument for *fill*
    for i in range(post.shape[0]):
      mask = _get_mask(data['time'][:,None],data[dir+'_std'][:,i],fill)
      post_sigma[i,mask] = np.inf
      post[i,mask] = np.nan
    
    out[dir] = post.T
    out[dir+'_std'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out['time'] = output_times
  out.check_self_consistency()
  return out
  

def pygeons_sgpr(data,sigma,cls,order=1,diff=(0,0),
                 procs=0,fill='none',
                 output_lonlat=None):
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

  output_lonlat : (N,2) array, optional
    Positions for the output data set, defaults to positions in the 
    input data set. 
    
  '''
  data.check_self_consistency()
  out = DataDict(data)

  # convert units of sigma from mm**p years**q to m**p days**q
  sigma *= 0.001**data['space_exponent']*365.25**data['time_exponent']
  # convert units of cls from km to m
  cls *= 1000.0  
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  # set output positions
  if output_lonlat is None:
    output_lonlat = np.array([data['longitude'],data['latitude']],copy=True).T
    output_id = np.array(data['id'],copy=True)
  else:  
    output_lonlat = np.asarray(output_lonlat)
    output_id = np.array(['%04d' % i for i in range(output_lonlat.shape[0])])

  output_x,output_y = bm(output_lonlat[:,0],output_lonlat[:,1])
  output_pos = np.array([output_x,output_y]).T
  # scaling factor for numerical stability
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.gpr.gpr(
                        pos,data[dir],data[dir+'_std'],(0.0,sigma**2,cls),
                        x=output_pos,basis=rbf.basis.ga,order=order,diff=diff,procs=procs)
    # loop over the times and mask the stations based on the 
    # argument for *fill*
    for i in range(post.shape[0]):
      mask = _get_mask(output_pos,data[dir+'_std'][i,:],fill)
      post_sigma[i,mask] = np.inf
      post[i,mask] = np.nan
    
    out[dir] = post
    out[dir+'_std'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  # set the new lon lat and id if output_lonlat was given
  out['longitude'] = output_lonlat[:,0]
  out['latitude'] = output_lonlat[:,1]
  out['id'] = output_id
  out.check_self_consistency()
  return out
  

def pygeons_tfilter(data,diff=(0,),fill='none',
                    break_dates=None,**kwargs):
  ''' 
  time smoothing
  '''
  data.check_self_consistency()
  vert,smp = make_time_vert_smp(break_dates)
  out = DataDict(data)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        data['time'][:,None],data[dir].T,
                        sigma=data[dir+'_std'].T,
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,
                        **kwargs)
    out[dir] = post.T
    out[dir+'_std'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out.check_self_consistency()
  return out


def pygeons_sfilter(data,diff=(0,0),fill='none',
                    break_lons=None,break_lats=None,
                    break_conn=None,**kwargs):
  ''' 
  space smoothing
  '''
  data.check_self_consistency()
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  vert,smp = make_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  out = DataDict(data)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        pos,data[dir],
                        sigma=data[dir+'_std'],
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,     
                        **kwargs)
    out[dir] = post
    out[dir+'_std'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  out.check_self_consistency()
  return out

