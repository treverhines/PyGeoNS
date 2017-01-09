''' 
Defines functions which are called by the PyGeoNS executables. These 
functions are for data cleaning.
'''
from __future__ import division
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pygeons.datadict import DataDict
from pygeons.mjd import mjd,mjd_inv
from pygeons.basemap import make_basemap
from pygeons.clean.iclean import interactive_cleaner
from pygeons.breaks import make_space_vert_smp
from pygeons.plot.plot import (_unit_string,
                               _unit_conversion,
                               _setup_map_ax,
                               _setup_ts_ax)                               
logger = logging.getLogger(__name__)


def pygeons_merge(data,radius):
  ''' 
  Merge stations that are within *radius* of eachother. 
  
  Parameters
  ----------
  data : dict
    Data dictionary
  
  radius : float
    Minimum distance between stations

  '''
  data.check_self_consistency()
  out = DataDict(data)  
  bm = make_basemap(data['longitude'],data['latitude'])
  while True:
    pos = np.array(bm(out['longitude'],out['latitude'])).T
    kd = cKDTree(pos)
    dist,idx = kd.query(pos,2)
    if not np.any(dist[:,1] < radius):
      break
    
    # find the two closest stations and merge them  
    idx1 = np.argmin(dist[:,1])
    idx2 = idx[idx1,1]
    print(pos[idx1])
    print(pos[idx2])
    out['longitude'][idx1] = 0.5*(out['longitude'][idx1] + out['longitude'][idx2])
    out['latitude'][idx1] = 0.5*(out['latitude'][idx1] + out['latitude'][idx2])
    for dir in ['east','north','vertical']:
      w1 = 1.0/out[dir+'_std'][:,idx1]**2
      w2 = 1.0/out[dir+'_std'][:,idx2]**2
      d1 = np.nan_to_num(out[dir][:,idx1])
      d2 = np.nan_to_num(out[dir][:,idx2])
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out[dir][:,idx1] = (d1*w1 + d2*w2)/(w1 + w2)
        out[dir+'_std'][:,idx1] = np.sqrt(1.0/(w1 + w2))
        
      #out[dir][:,idx1] = ((out[dir][:,idx1]/out[dir+'_std'][:,idx1]**2 + 
      #                     out[dir][:,idx2]/out[dir+'_std'][:,idx2]**2)/
      #                    (1.0/out[dir+'_std'][:,idx1]**2 + 
      #                     1.0/out[dir+'_std'][:,idx2]**2))
      #out[dir+'_std'][:,idx1] = 1.0/np.sqrt(1.0/out[dir+'_std'][:,idx1]**2 + 
      #                                      1.0/out[dir+'_std'][:,idx2]**2)

    out['longitude'] = np.delete(out['longitude'],idx2)
    out['latitude'] = np.delete(out['latitude'],idx2)
    out['id'] = np.delete(out['id'],idx2)
    for dir in ['east','north','vertical']:
      out[dir] = np.delete(out[dir],idx2,axis=1)
      out[dir+'_std'] = np.delete(out[dir+'_std'],idx2,axis=1)
    
  out.check_self_consistency()
  return out  

def pygeons_crop(data,start_date=None,stop_date=None,
                 min_lat=-np.inf,max_lat=np.inf,
                 min_lon=-np.inf,max_lon=np.inf):
  ''' 
  Sets the time span of the data set to be between *start_date* and 
  *stop_date*.
  
  Parameters
  ----------
  data : dict
    data dictionary
      
  start_date : str, optional
    start date of output data set in YYYY-MM-DD. Uses the start date 
    of *data* if not provided. Defaults to the earliest date.

  stop_date : str, optional
    Stop date of output data set in YYYY-MM-DD. Uses the stop date 
    of *data* if not provided. Defaults to the latest date.
      
  min_lon, max_lon, min_lat, max_lat : float, optional
    Spatial bounds on the output data set
    
  Returns
  -------
  out_dict : dict
    output data dictionary

  '''
  data.check_self_consistency()
  if start_date is None:
    start_date = mjd_inv(data['time'].min(),'%Y-%m-%d')
  if stop_date is None:
    stop_date = mjd_inv(data['time'].max(),'%Y-%m-%d')
  
  out = DataDict(data) # make copy of the data dictionary

  # remove times that are not within the bounds of *start_date* and 
  # *stop_date*
  start_time = int(mjd(start_date,'%Y-%m-%d'))
  stop_time = int(mjd(stop_date,'%Y-%m-%d'))
  idx = ((data['time'] >= start_time) &
         (data['time'] <= stop_time))
  out['time'] = out['time'][idx]         
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][idx,:]
    out[dir + '_std'] = out[dir + '_std'][idx,:]
    
  # remove stations that are not within the bounds 
  idx = ((data['longitude'] > min_lon) &
         (data['longitude'] < max_lon) &
         (data['latitude'] > min_lat) &
         (data['latitude'] < max_lat))
         
  out['id'] = out['id'][idx]
  out['longitude'] = out['longitude'][idx]
  out['latitude'] = out['latitude'][idx]
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][:,idx]
    out[dir + '_std'] = out[dir + '_std'][:,idx]
    
  # make a boolean array indicating whether data is available for a 
  # times and stations
  is_missing = (~np.isfinite(out['east_std']) &
                ~np.isfinite(out['north_std']) &
                ~np.isfinite(out['vertical_std']))

  # find and remove times that have no observations                
  idx = ~np.all(is_missing,axis=1)
  out['time'] = out['time'][idx]         
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][idx,:]
    out[dir + '_std'] = out[dir + '_std'][idx,:]
    
  # find and remove stations that have no observations                
  idx = ~np.all(is_missing,axis=0)
  out['id'] = out['id'][idx]
  out['longitude'] = out['longitude'][idx]
  out['latitude'] = out['latitude'][idx]
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][:,idx]
    out[dir + '_std'] = out[dir + '_std'][:,idx]
  
  out.check_self_consistency()
  return out


def pygeons_clean(data,resolution='i',
                  break_lons=None,break_lats=None,
                  break_conn=None,**kwargs):
  ''' 
  runs the PyGeoNS Interactive Cleaner
  
  Parameters
  ----------
    data : dict
      data dictionary

    resolution : str
      basemap resolution    
      
    **kwargs : 
      gets passed to pygeons.clean.clean
         
  Returns
  -------
    out : dict
      output data dictionary 
    
  '''
  data.check_self_consistency()
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax,data['time'])
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(data['longitude'],data['latitude'],
                    resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = make_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  t = data['time']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]

  conv = _unit_conversion(data['space_exponent'],
                          data['time_exponent'])
  units = _unit_string(data['space_exponent'],
                       data['time_exponent'])

  u = conv*data['east']
  v = conv*data['north']
  z = conv*data['vertical']
  su = conv*data['east_std']
  sv = conv*data['north_std']
  sz = conv*data['vertical_std']
  clean_data = interactive_cleaner(
                 t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
                 map_ax=map_ax,ts_ax=ts_ax,
                 time_labels=dates,
                 units=units,
                 station_labels=data['id'],
                 **kwargs)

  out = DataDict(data)
  out['east'] = clean_data[0]/conv
  out['north'] = clean_data[1]/conv
  out['vertical'] = clean_data[2]/conv
  out['east_std'] = clean_data[3]/conv
  out['north_std'] = clean_data[4]/conv
  out['vertical_std'] = clean_data[5]/conv
  out.check_self_consistency()
  return out


