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
from pygeons.datadict import DataDict
from pygeons.basemap import make_basemap
from pygeons.breaks import make_time_vert_smp, make_space_vert_smp
logger = logging.getLogger(__name__)


def pygeons_tgpr(data,var,cls,order=1,diff=(0,),
                 procs=0,fill='none'):
  ''' 
  Temporal Gaussian process regression
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  var : float
    Prior variance 
  
  cls : float
    Characteristic length-scale in meters
  
  order : int, optional
    Order of the polynomial null space
  
  diff : int, optional
    Derivative order
  
  procs : int, optional
    Number of subprocesses to spawn       

  '''
  data.check_self_consistency()
  out = DataDict(data)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.gpr.gpr(
                        data['time'][:,None],
                        data[dir].T,
                        data[dir+'_std'].T,
                        (0.0,var,cls),
                        basis=rbf.basis.ga,
                        order=order,
                        diff=diff,
                        fill=fill,
                        procs=procs)
    out[dir] = post.T
    out[dir+'_std'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out.check_self_consistency()
  return out
  
  gp = PriorGaussianProcess(rbf.basis.ga,(0.0,var,cls),order=order)

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

