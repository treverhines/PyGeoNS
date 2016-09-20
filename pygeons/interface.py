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
  

def _make_time_breaks(break_dates):
  if break_dates is None: break_dates = []
  # subtract half a day to get rid of any ambiguity about what day 
  # the dislocation is observed
  breaks = [decday(d,'%Y-%m-%d') - 0.5 for d in break_dates]   
  vert = np.array(breaks).reshape((-1,1))
  smp = np.arange(vert.shape[0]).reshape((-1,1))  
  return vert,smp


def _make_space_breaks(end1_lons,end1_lats,end2_lons,end2_lats,bm):
  if end1_lons is None: end1_lons = []
  if end1_lats is None: end1_lats = []
  if end2_lons is None: end2_lons = []
  if end2_lats is None: end2_lats = []
  end1 = [bm(i,j) for i,j in zip(end1_lons,end1_lats)]
  end2 = [bm(i,j) for i,j in zip(end2_lons,end2_lats)]
  end1 = np.array(end1).reshape((-1,2))
  end2 = np.array(end2).reshape((-1,2))
  breaks = np.concatenate((end1[:,None,:],end2[:,None,:]),axis=1)
  vert = np.array(breaks).reshape((-1,2))
  smp = np.arange(vert.shape[0]).reshape((-1,2))  
  return vert,smp


def _check_io(fin):
  ''' 
  checks the input and output for functions that take and return data dictionaries
  '''
  @wraps(fin)
  def fout(data,*args,**kwargs):
    data.check_self_consistency()
    data_out = fin(data,*args,**kwargs)
    data_out.check_self_consistency()
    return data_out

  return fout  

  
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
  returns the meridians and parallels that should be plotted.
  '''
  diff_lon = (bm.urcrnrlon-bm.llcrnrlon)
  round_digit = int(np.ceil(np.log10(ticks/diff_lon)))
  dlon = np.round(diff_lon/ticks,round_digit)

  diff_lat = (bm.urcrnrlat-bm.llcrnrlat)
  round_digit = int(np.ceil(np.log10(ticks/diff_lat)))
  dlat = np.round(diff_lat/ticks,round_digit)

  meridians = np.arange(np.floor(bm.llcrnrlon),
                        np.ceil(bm.urcrnrlon),dlon)
  parallels = np.arange(np.floor(bm.llcrnrlat),
                        np.ceil(bm.urcrnrlat),dlat)

  return meridians,parallels
  

def _setup_map_ax(bm,ax):
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


@_check_io
def tfilter(data,
            cutoff=None,
            diff=(0,),
            fill='none',
            procs=0,
            samples=100,
            break_dates=None):
  ''' 
  time smoothing
  '''
  vert,smp = _make_time_breaks(break_dates)
  out = DataDict(data)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        data['time'][:,None],data[dir].T,
                        sigma=data[dir+'_std'].T,
                        cutoff=cutoff,
                        diffs=diff,
                        procs=procs,
                        samples=samples,
                        vert=vert,smp=smp,
                        fill=fill)
    out[dir] = post.T
    out[dir+'_std'] = post_sigma.T

  # set the time units
  out['time_power'] -= sum(diff)
  return out


@_check_io
def sfilter(data,
            cutoff=None,
            diff=(0,0),
            fill='none',
            procs=0,
            samples=100,
            break_lons1=None,break_lats1=None,
            break_lons2=None,break_lats2=None):
  ''' 
  space smoothing
  '''
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  vert,smp = _make_space_breaks(break_lons1,break_lats1,
                                break_lons2,break_lats2,bm)
  out = DataDict(data)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        pos,data[dir],
                        sigma=data[dir+'_std'],
                        cutoff=cutoff,
                        diffs=diff,
                        procs=procs,
                        samples=samples,
                        vert=vert,smp=smp,         
                        fill=fill)
    out[dir] = post
    out[dir+'_std'] = post_sigma

  # set the space units
  out['space_power'] -= sum(diff)
  return out


@_check_io
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
  vert,smp = _make_time_breaks(break_dates)
      
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

  return out


@_check_io
def clean(data,resolution='i',
          break_lons1=None,break_lats1=None,
          break_lons2=None,break_lats2=None,
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
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax,data['time'])
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = _make_space_breaks(break_lons1,break_lats1,
                                break_lons2,break_lats2,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

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
  out['east'] = clean_data[0]
  out['north'] = clean_data[1]
  out['vertical'] = clean_data[2]
  out['east_std'] = clean_data[3]
  out['north_std'] = clean_data[4]
  out['vertical_std'] = clean_data[5]
  return out


def view(data_list,resolution='i',
         break_lons1=None,break_lats1=None,
         break_lons2=None,break_lats2=None,
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

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax,data_list[0]['time'])
   
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw breaks if there are any
  vert,smp = _make_space_breaks(break_lons1,break_lats1,
                                break_lons2,break_lats2,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

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
           break_lons1=None,break_lats1=None,
           break_lons2=None,break_lats2=None,
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
  bm = _make_basemap(lon,lat,
                     resolution=resolution)
  _setup_map_ax(bm,ax)

  # draw breaks if there are any
  vert,smp = _make_space_breaks(break_lons1,break_lats1,
                                break_lons2,break_lats2,bm)
  for s in smp:
    ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

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
