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
  time_power : int
  space_power : int

'''
from __future__ import division
import numpy as np
import pygeons.view
import pygeons.strain
import pygeons.clean
import rbf.filter
import logging
import matplotlib.pyplot as plt
from pygeons.mean import MeanInterpolant
from pygeons.datadict import DataDict
from pygeons.dateconv import decday_inv
from pygeons.dateconv import decday
from mpl_toolkits.basemap import Basemap
from functools import wraps
logger = logging.getLogger(__name__)


def _unit_string(space_power,time_power):
  ''' 
  returns a string indicating the units
  '''
  if (space_power == 0) & (time_power == 0):
    return ''

  if space_power == 0:
    space_str = '1'
  elif space_power == 1:
    space_str = 'mm'
  else:
    space_str = 'mm^%s' % space_power

  if time_power == 0:
    time_str = ''
  elif time_power == -1:
    time_str = '/yr'
  else:
    time_str = '/yr^%s' % -time_power
  
  return space_str + time_str
        

def _unit_conversion(space_power,time_power):
  ''' 
  returns the scalar which converts 
  
    meters**(space_power) * days*(time_power)
  
  to   

    mm**(space_power) * years*(time_power)
  '''
  return 1000**space_power * (1.0/365.25)**time_power
  

def _get_time_vert_smp(break_dates):
  ''' 
  returns the vertices and simplices defining the time breaks
  ''' 
  if break_dates is None: break_dates = []
  # subtract half a day to get rid of any ambiguity about what day 
  # the dislocation is observed
  breaks = [decday(d,'%Y-%m-%d') - 0.5 for d in break_dates]   
  vert = np.array(breaks).reshape((-1,1))
  smp = np.arange(vert.shape[0]).reshape((-1,1))  
  return vert,smp


def _get_space_vert_smp(break_lons,break_lats,break_conn,bm):
  ''' 
  returns the vertices and simplices defining the space breaks
  
  Parameters
  ----------
  break_lons : (N,) float array
    
  break_lats : (N,) float array
      
  break_conn : (M,) str array 
  ''' 
  if break_lons is None:
    break_lons = np.zeros(0,dtype=float)
  else:
    break_lons = np.asarray(break_lons,dtype=float)

  if break_lats is None:
    break_lats = np.zeros(0,dtype=float)
  else:
    break_lats = np.asarray(break_lats,dtype=float)
    
  if break_lons.shape[0] != break_lats.shape[0]:
    raise ValueError('*break_lons* and *break_lats* must have the same length')
    
  N = break_lons.shape[0]
  if break_conn is None:
    if N != 0:
      break_conn = ['-'.join(np.arange(N).astype(str))]
    else:
      break_conn = []
        
  smp = [] 
  for c in break_conn:
    idx = np.array(c.split('-'),dtype=int)  
    smp += zip(idx[:-1],idx[1:])    
    
  smp = np.array(smp,dtype=int).reshape((-1,2))
  vert = [bm(i,j) for i,j in zip(break_lons,break_lats)]
  vert = np.array(vert,dtype=float).reshape((-1,2))
  return vert,smp


def _make_basemap(lon,lat,resolution=None):
  ''' 
  creates a transverse mercator projection which is centered about the 
  given positions
  '''
  lon_buff = max(0.1,lon.ptp()/20.0)
  lat_buff = max(0.1,lat.ptp()/20.0)
  llcrnrlon = min(lon) - lon_buff
  llcrnrlat = min(lat) - lat_buff
  urcrnrlon = max(lon) + lon_buff
  urcrnrlat = max(lat) + lat_buff
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 resolution=resolution, 
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat)


def _get_meridians_and_parallels(bm,ticks):
  ''' 
  attempts to find nice locations for the meridians and parallels 
  '''
  diff_lon = (bm.urcrnrlon-bm.llcrnrlon)
  round_digit = int(np.ceil(np.log10(ticks/diff_lon)))
  dlon = np.round(diff_lon/ticks,round_digit)

  diff_lat = (bm.urcrnrlat-bm.llcrnrlat)
  round_digit = int(np.ceil(np.log10(ticks/diff_lat)))
  dlat = np.round(diff_lat/ticks,round_digit)

  # round down to the nearest rounding digit
  lon_low = np.floor(bm.llcrnrlon*10**round_digit)/10**round_digit
  lat_low = np.floor(bm.llcrnrlat*10**round_digit)/10**round_digit
  # round up to the nearest rounding digit
  lon_high = np.ceil(bm.urcrnrlon*10**round_digit)/10**round_digit
  lat_high = np.ceil(bm.urcrnrlat*10**round_digit)/10**round_digit

  meridians = np.arange(lon_low,lon_high+dlon,dlon)
  parallels = np.arange(lat_low,lat_high+dlat,dlat)
  return meridians,parallels
  

def _setup_map_ax(bm,ax):
  ''' 
  prepares the map axis for display
  '''
  # function which prints out the coordinates on the bottom left 
  # corner of the figure
  def coord_string(x,y):                         
    str = 'x : %g  y : %g  ' % (x,y)
    str += '(lon : %g E  lat : %g N)' % bm(x,y,inverse=True)
    return str 

  ax.format_coord = coord_string
  bm.drawcountries(ax=ax)
  bm.drawstates(ax=ax) 
  bm.drawcoastlines(ax=ax)
  mer,par =  _get_meridians_and_parallels(bm,3)
  bm.drawmeridians(mer,
                   labels=[0,0,0,1],dashes=[2,2],
                   ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))
  bm.drawparallels(par,
                   labels=[1,0,0,0],dashes=[2,2],
                   ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))
  return
                     

def _setup_ts_ax(ax_lst,times):
  ''' 
  prepares the time series axes for display
  '''
  # display time in decday and date on time series plot
  def ts_coord_string(x,y):                         
    str = 'time : %g  ' % x
    str += '(date : %s)' % decday_inv(x,'%Y-%m-%d')
    return str 

  ticks = np.linspace(times.min(),times.max(),13)[1:-1:2]
  ticks = np.round(ticks)
  tick_labels = [decday_inv(t,'%Y-%m-%d') for t in ticks]
  ax_lst[2].set_xticks(ticks)
  ax_lst[2].set_xticklabels(tick_labels)
  ax_lst[0].format_coord = ts_coord_string
  ax_lst[1].format_coord = ts_coord_string
  ax_lst[2].format_coord = ts_coord_string
  return


def tfilter(data,
            diff=(0,),
            fill='none',
            break_dates=None,
            **kwargs):
  ''' 
  time smoothing
  '''
  data.check_self_consistency()
  vert,smp = _get_time_vert_smp(break_dates)
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
  out['time_power'] -= sum(diff)
  out.check_self_consistency()
  return out


def sfilter(data,
            diff=(0,0),
            fill='none',
            break_lons=None,
            break_lats=None,
            break_conn=None,
            **kwargs):
  ''' 
  space smoothing
  '''
  data.check_self_consistency()
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  vert,smp = _get_space_vert_smp(break_lons,break_lats,
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
  out['space_power'] -= sum(diff)
  out.check_self_consistency()
  return out


def downsample(data,sample_period=1,start_date=None,stop_date=None,
               break_dates=None):
  ''' 
  downsamples the data set along the time axis
  
  Parameters
  ----------
    data : dict
      data dictionary
      
    sample_period : int, optional
      sample period of the output data set in days. Output data is 
      computed using a running mean with this width. This should be an 
      odd integer in order to avoid double counting the observations 
      at some days. Defaults to 1.
      
    start_date : str, optional
      start date of output data set in YYYY-MM-DD. Uses the start date 
      of *data* if not provided

    stop_date : str, optional
      stop date of output data set in YYYY-MM-DD. Uses the stop date 
      of *data* if not provided

    break_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed
      
  Returns
  -------
    out_dict : dict
      output data dictionary

  '''
  data.check_self_consistency()
  vert,smp = _get_time_vert_smp(break_dates)
      
  # if the start and stop time are not specified then use the min and 
  # max times
  if start_date is None:
    start_date = decday_inv(data['time'].min(),'%Y-%m-%d')
  if stop_date is None:
    stop_date = decday_inv(data['time'].max(),'%Y-%m-%d')
  
  start_time = int(decday(start_date,'%Y-%m-%d'))
  stop_time = int(decday(stop_date,'%Y-%m-%d'))
  sample_period = int(sample_period)
  time_itp = np.arange(start_time,stop_time+1,sample_period)
  out = DataDict(data)
  out['time'] = time_itp
  for dir in ['east','north','vertical']:
    mi = MeanInterpolant(data['time'][:,None],
                         data[dir].T,sigma=data[dir+'_std'].T,
                         vert=vert,smp=smp)
    post,post_sigma = mi(time_itp[:,None])
    out[dir] = post.T
    out[dir+'_std'] = post_sigma.T

  out.check_self_consistency()
  return out


def clean(data,resolution='i',
          break_lons=None,break_lats=None,
          break_conn=None,
          **kwargs):
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
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = _get_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  t = data['time']
  dates = [decday_inv(ti,'%Y-%m-%d') for ti in t]

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
  clean_data = pygeons.clean.interactive_cleaner(
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


def view(data_list,resolution='i',
         break_lons=None,break_lats=None,
         break_conn=None,
         **kwargs):
  ''' 
  runs the PyGeoNS interactive Viewer
  
  Parameters
  ----------
    data_list : (N,) list of dicts
      list of data dictionaries being plotted
      
    resolution : str
      basemap resolution
      
    **kwargs :
      gets passed to pygeons.view.view

  '''
  for d in data_list:
    d.check_self_consistency()
  for d in data_list[1:]:
    d.check_compatibility(data_list[0])

  t = data_list[0]['time']
  lon = data_list[0]['longitude']
  lat = data_list[0]['latitude']
  id = data_list[0]['id']
  dates = [decday_inv(ti,'%Y-%m-%d') for ti in t]

  conv = _unit_conversion(data_list[0]['space_power'],
                          data_list[0]['time_power'])
  units = _unit_string(data_list[0]['space_power'],
                       data_list[0]['time_power'])

  u = [conv*d['east'] for d in data_list]
  v = [conv*d['north'] for d in data_list]
  z = [conv*d['vertical'] for d in data_list]
  su = [conv*d['east_std'] for d in data_list]
  sv = [conv*d['north_std'] for d in data_list]
  sz = [conv*d['vertical_std'] for d in data_list]

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax,data_list[0]['time'])
   
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw breaks if there are any
  vert,smp = _get_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.view.interactive_viewer(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    ts_ax=ts_ax,map_ax=map_ax,
    station_labels=id,
    time_labels=dates,
    units=units,
    **kwargs)

  return

def strain(data_dx,data_dy,resolution='i',
           break_lons=None,break_lats=None,
           break_conn=None,
           **kwargs):
  ''' 
  runs the PyGeoNS Interactive Strain Viewer
  
  Parameters
  ----------
    data_dx : x derivative data dictionaries 

    data_dy : y derivative data dictionaries 
      
    resolution : str
      basemap resolution
      
    **kwargs :
      gets passed to pygeons.strain.view

  '''
  data_dx.check_self_consistency()
  data_dy.check_self_consistency()
  data_dx.check_compatibility(data_dy)
  if (data_dx['space_power'] != 0) | data_dy['space_power'] != 0:
    raise ValueError('data sets cannot have spatial units')
  
  t = data_dx['time']
  lon = data_dx['longitude']
  lat = data_dx['latitude']
  dates = [decday_inv(ti,'%Y-%m-%d') for ti in t]

  conv = _unit_conversion(data_dx['space_power'],
                          data_dx['time_power'])
  units = _unit_string(data_dx['space_power'],
                       data_dx['time_power'])

  ux = conv*data_dx['east']
  sux = conv*data_dx['east_std']
  vx = conv*data_dx['north']
  svx = conv*data_dx['north_std']
  uy = conv*data_dy['east']
  suy = conv*data_dy['east_std']
  vy = conv*data_dy['north']
  svy = conv*data_dy['north_std']
   
  fig,ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,resolution=resolution)
  _setup_map_ax(bm,ax)

  # draw breaks if there are any
  vert,smp = _get_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for s in smp:
    ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.strain.interactive_strain_viewer(
    t,pos,
    ux,uy,vx,vy,
    sux,suy,svx,svy,
    ax=ax,
    time_labels=dates,
    units=units,
    **kwargs)

  return
