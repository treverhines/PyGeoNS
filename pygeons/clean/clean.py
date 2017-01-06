''' 
Defines functions which are called by the PyGeoNS executables. These 
functions are for data cleaning.
'''
from __future__ import division
import numpy as np
import logging
import matplotlib.pyplot as plt
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


def pygeons_crop(data,start_date=None,stop_date=None):
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
    stop date of output data set in YYYY-MM-DD. Uses the stop date 
    of *data* if not provided. Defaults to the latest date.
      
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
        
  start_time = int(mjd(start_date,'%Y-%m-%d'))
  stop_time = int(mjd(stop_date,'%Y-%m-%d'))
  idx = ((data['time'] >= start_time) &
         (data['time'] <= stop_time))

  out = DataDict(data) # make copy of the data dictionary
  out['time'] = out['time'][idx]         
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][idx,:]
    out[dir + '_std'] = out[dir + '_std'][idx,:]
    
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

  conv = _unit_conversion(data['space_power'],
                          data['time_power'])
  units = _unit_string(data['space_power'],
                       data['time_power'])

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


