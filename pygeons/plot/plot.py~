''' 
Defines functions which are called by the PyGeoNS executables. These 
are the highest level of plotting functions. There is a vector 
plotting function and a strain plotting function. Both take data 
dictionaries as input, as well as additional plotting parameters.
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.ticker import FuncFormatter,MaxNLocator
from pygeons.plot.iview import interactive_viewer
from pygeons.plot.istrain import interactive_strain_viewer
from pygeons.mjd import mjd_inv
from pygeons.datacheck import check_data,check_compatibility
from pygeons.basemap import make_basemap
from pygeons.breaks import make_space_vert_smp
logger = logging.getLogger(__name__)


def _unit_string(space_exponent,time_exponent):
  ''' 
  returns a string indicating the units
  '''
  if (space_exponent == 0) & (time_exponent == 0):
    return ''

  if space_exponent == 0:
    space_str = '1'
  elif space_exponent == 1:
    space_str = 'mm'
  else:
    space_str = 'mm^%s' % space_exponent

  if time_exponent == 0:
    time_str = ''
  elif time_exponent == -1:
    time_str = '/yr'
  else:
    time_str = '/yr^%s' % -time_exponent
  
  return space_str + time_str
        

def _unit_conversion(space_exponent,time_exponent):
  ''' 
  returns the scalar which converts 
  
    meters**(space_exponent) * days*(time_exponent)
  
  to   

    mm**(space_exponent) * years*(time_exponent)
  '''
  return 1000**space_exponent * (1.0/365.25)**time_exponent
  

def _get_meridians_and_parallels(bm,ticks):
  ''' 
  attempts to find nice locations for the meridians and parallels 
  based on the current axis limits
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
  def coord_formatter(x,y):                         
    out = 'x : %g  y : %g  ' % (x,y)
    out += '(lon : %g E  lat : %g N)' % bm(x,y,inverse=True)
    return out

  ax.format_coord = coord_formatter
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
                     

def _setup_ts_ax(ax_lst):
  ''' 
  prepares the time series axes for display
  '''
  # display time in MJD and date on time series plot
  def coord_formatter(x,y):                         
    ''' 
    Takes coordinates, *x* and *y*, and returns their string 
    representation
    '''
    out = 'time : %g  ' % x
    out += '(date : %s)' % mjd_inv(x,'%Y-%m-%d')
    return out

  @FuncFormatter
  def xtick_formatter(x,p):
    ''' 
    Takes *x* and the number of ticks, *p*, and returns a string 
    representation of *x*
    '''
    try:
      out = mjd_inv(x,'%Y-%m-%d')
    except (ValueError,OverflowError):
      out = ''  
      
    return out
    
  for a in ax_lst: a.get_xaxis().set_major_formatter(xtick_formatter)
  for a in ax_lst: a.get_xaxis().set_major_locator(MaxNLocator(6))
  for a in ax_lst: a.format_coord = coord_formatter
  return


def pygeons_view(data_list,resolution='i',
                 break_lons=None,break_lats=None,
                 break_conn=None,**kwargs):
  ''' 
  runs the PyGeoNS interactive Viewer
  
  Parameters
  ----------
    data_list : (N,) list of dicts
      list of data dictionaries being plotted
      
    resolution : str
      basemap resolution
    
    break_lons : (N,) array

    break_lats : (N,) array

    break_con : (M,) array   
      
    **kwargs :
      gets passed to pygeons.plot.view.interactive_view

  '''
  for d in data_list: check_data(d)
  for d in data_list[1:]: check_compatibility(data_list[0],d)

  t = data_list[0]['time']
  lon = data_list[0]['longitude']
  lat = data_list[0]['latitude']
  id = data_list[0]['id']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  conv = _unit_conversion(data_list[0]['space_exponent'],
                          data_list[0]['time_exponent'])
  units = _unit_string(data_list[0]['space_exponent'],
                       data_list[0]['time_exponent'])
  u = [conv*d['east'] for d in data_list]
  v = [conv*d['north'] for d in data_list]
  z = [conv*d['vertical'] for d in data_list]
  su = [conv*d['east_std'] for d in data_list]
  sv = [conv*d['north_std'] for d in data_list]
  sz = [conv*d['vertical_std'] for d in data_list]
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax)
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(lon,lat,resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = make_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  interactive_viewer(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    ts_ax=ts_ax,map_ax=map_ax,
    station_labels=id,time_labels=dates,
    units=units,**kwargs)

  return


def pygeons_strain(data_dx,data_dy,resolution='i',
                   break_lons=None,break_lats=None,
                   break_conn=None,**kwargs):
  ''' 
  runs the PyGeoNS Interactive Strain Viewer
  
  Parameters
  ----------
    data_dx : x derivative data dictionaries 

    data_dy : y derivative data dictionaries 
      
    resolution : str
      basemap resolution
      
    break_lons : (N,) array

    break_lats : (N,) array

    break_con : (M,) array   

    **kwargs :
      gets passed to pygeons.strain.view

  '''
  check_data(data_dx)
  check_data(data_dy)
  check_compatibility(data_dx,data_dy)
  
  if (data_dx['space_exponent'] != 0) | data_dy['space_exponent'] != 0:
    raise ValueError('data sets cannot have spatial units')
  
  t = data_dx['time']
  id = data_dx['id']
  lon = data_dx['longitude']
  lat = data_dx['latitude']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  conv = _unit_conversion(data_dx['space_exponent'],
                          data_dx['time_exponent'])
  units = _unit_string(data_dx['space_exponent'],
                       data_dx['time_exponent'])
  exx = conv*data_dx['east'] 
  eyy = conv*data_dy['north']
  exy = 0.5*conv*(data_dx['north'] + data_dy['east'])
  sxx = conv*data_dx['east_std']
  syy = conv*data_dy['north_std']
  sxy = 0.5*conv*np.sqrt(data_dx['north_std']**2 + data_dy['east_std']**2)
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(lon,lat,resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = make_space_vert_smp(break_lons,break_lats,break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  ## draw map scale
  # find point 0.1,0.1
  # XXXXXXXXXXXX 
  scale_lon,scale_lat = bm(*map_ax.transData.inverted().transform(map_ax.transAxes.transform([0.15,0.1])),inverse=True)
  bm.drawmapscale(scale_lon,scale_lat,scale_lon,scale_lat,150,ax=map_ax,barstyle='fancy',fontsize=10)
  # XXXXXXXXXXXX 
  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax)
  interactive_strain_viewer(
    t,pos,exx,eyy,exy,sxx=sxx,syy=syy,sxy=sxy,
    map_ax=map_ax,ts_ax=ts_ax,time_labels=dates,
    station_labels=id,units=units,**kwargs)

  return
