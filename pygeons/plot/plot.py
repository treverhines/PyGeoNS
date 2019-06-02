''' 
Defines functions which are called by the PyGeoNS executables. These 
are the highest level of plotting functions. There is a vector 
plotting function and a strain plotting function. Both take data 
dictionaries as input, as well as additional plotting parameters.
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import logging
from matplotlib.ticker import FuncFormatter,MaxNLocator
from pygeons.io.convert import dict_from_hdf5
from pygeons.plot.ivector import interactive_vector_viewer,one_sigfig
from pygeons.plot.istrain import interactive_strain_viewer
from pygeons.mjd import mjd_inv
from pygeons.basemap import make_basemap
from pygeons.units import unit_conversion
from pygeons.io.io import _common_context
logger = logging.getLogger(__name__)


def _unit_string(space_exponent,time_exponent):
  ''' 
  returns a string indicating the units
  '''
  if space_exponent == 0:
    # if the space exponent is 0 then use units of microstrain
    space_str = '1e-6'
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
  

def _setup_map_ax(ax):
  ''' 
  prepares the map axis for display
  '''
  proj = ax.projection
  # function which prints out the coordinates on the bottom left 
  # corner of the figure
  def coord_formatter(x,y):                         
    lon, lat = proj.as_geodetic().transform_point(x, y, proj)
    out = 'x : %g  y : %g  ' % (x,y)
    out += '(lon : %g E  lat : %g N)' % (lon, lat)
    return out

  ax.format_coord = coord_formatter
  #bm.drawcountries(ax=ax,zorder=2)
  #bm.drawstates(ax=ax,zorder=2) 
  #bm.drawcoastlines(ax=ax,zorder=2)
  #mer,par =  _get_meridians_and_parallels(bm,3)
  #bm.drawmeridians(mer,
  #                 labels=[0,0,0,1],dashes=[2,2],
  #                 ax=ax,zorder=2,color=(0.3,0.3,0.3,1.0))
  #bm.drawparallels(par,
  #                 labels=[1,0,0,0],dashes=[2,2],
  #                 ax=ax,zorder=2,color=(0.3,0.3,0.3,1.0))
  #bm.drawmapboundary(ax=ax,fill_color=(0.9,0.9,0.9),zorder=2)
  #bm.fillcontinents(ax=ax,color=(1.0,1.0,1.0),lake_color=(0.9,0.9,0.9),zorder=0)
  #scale_lon,scale_lat = bm(*ax.transData.inverted().transform(ax.transAxes.transform([0.15,0.1])),
  #                         inverse=True)
  #scale_size = one_sigfig((bm.urcrnrx - bm.llcrnrx)/5.0)/1000.0
  #bm.drawmapscale(scale_lon,scale_lat,scale_lon,scale_lat,scale_size,
  #                ax=ax,barstyle='fancy',fontsize=10,zorder=2)
  return
                     

def _setup_ts_ax(ax_lst,maxn=6):
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
  for a in ax_lst: a.get_xaxis().set_major_locator(MaxNLocator(maxn))
  for a in ax_lst: a.format_coord = coord_formatter
  return


def pygeons_vector_view(input_files,map_resolution='i',**kwargs):
  ''' 
  runs the PyGeoNS interactive vector viewer
  
  Parameters
  ----------
    data_list : (N,) list of dicts
      list of data dictionaries being plotted
      
    map_resolution : str
      basemap resolution
    
    **kwargs :
      gets passed to pygeons.plot.view.interactive_view

  '''
  logger.info('Running pygeons vector-view ...')
  data_list = [dict_from_hdf5(i) for i in input_files]
  data_list = _common_context(data_list)
  
  # use filenames for dataset labels if none were provided
  dataset_labels = kwargs.pop('dataset_labels',input_files)

  t = data_list[0]['time']
  lon = data_list[0]['longitude']
  lat = data_list[0]['latitude']
  id = data_list[0]['id']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  units = _unit_string(data_list[0]['space_exponent'],
                       data_list[0]['time_exponent'])
  # factor that converts units of days and m to the units in *units*
  conv = 1.0/unit_conversion(units,time='day',space='m')
  u = [conv*d['east'] for d in data_list]
  v = [conv*d['north'] for d in data_list]
  z = [conv*d['vertical'] for d in data_list]
  su = [conv*d['east_std_dev'] for d in data_list]
  sv = [conv*d['north_std_dev'] for d in data_list]
  sz = [conv*d['vertical_std_dev'] for d in data_list]
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax)
  proj = make_basemap(lon,lat)
  map_fig,map_ax = plt.subplots(num='Map View',
                                facecolor='white',
                                subplot_kw={'projection':proj})
  _setup_map_ax(map_ax)
  pos = proj.transform_points(proj.as_geodetic(), lon, lat)[:,:2]
  interactive_vector_viewer(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    ts_ax=ts_ax,map_ax=map_ax,
    dataset_labels=dataset_labels,
    station_labels=id,time_labels=dates,
    units=units,**kwargs)

  return


def pygeons_strain_view(xdiff_file,ydiff_file,map_resolution='i',**kwargs):
  ''' 
  runs the PyGeoNS Interactive Strain Viewer
  
  Parameters
  ----------
    xdiff_file : str

    ydiff_file : str
      
    map_resolution : str
      basemap resolution
      
    **kwargs :
      gets passed to pygeons.strain.view

  '''
  logger.info('Running pygeons strain-view ...')
  data_dx = dict_from_hdf5(xdiff_file)  
  data_dy = dict_from_hdf5(ydiff_file)  
  data_dx,data_dy = _common_context([data_dx,data_dy])
  
  if ((data_dx['space_exponent'] != 0) | 
      (data_dy['space_exponent'] != 0)):
    raise ValueError('The input datasets cannot have spatial units')
  
  t = data_dx['time']
  id = data_dx['id']
  lon = data_dx['longitude']
  lat = data_dx['latitude']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  units = _unit_string(data_dx['space_exponent'],
                       data_dx['time_exponent'])
  # factor that converts units of days and m to the units in *units*
  conv = 1.0/unit_conversion(units,time='day',space='m')
  exx = conv*data_dx['east'] 
  eyy = conv*data_dy['north']
  exy = 0.5*conv*(data_dx['north'] + data_dy['east'])
  sxx = conv*data_dx['east_std_dev']
  syy = conv*data_dy['north_std_dev']
  sxy = 0.5*conv*np.sqrt(data_dx['north_std_dev']**2 + data_dy['east_std_dev']**2)

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax)
  proj = make_basemap(lon,lat)
  map_fig,map_ax = plt.subplots(num='Map View',
                                facecolor='white',
                                subplot_kw={'projection':proj})
  _setup_map_ax(map_ax)
  pos = proj.transform_points(proj.as_geodetic(), lon, lat)[:,:2]
  interactive_strain_viewer(
    t,pos,exx,eyy,exy,sxx=sxx,syy=syy,sxy=sxy,
    map_ax=map_ax,ts_ax=ts_ax,time_labels=dates,
    station_labels=id,units=units,**kwargs)

  return
