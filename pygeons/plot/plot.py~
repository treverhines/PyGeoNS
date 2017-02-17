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
from pygeons.plot.iview import interactive_viewer,one_sigfig
from pygeons.plot.istrain import interactive_strain_viewer
from pygeons.mjd import mjd_inv
from pygeons.basemap import make_basemap
from pygeons.breaks import make_space_vert_smp
logger = logging.getLogger(__name__)


def _common_context(data_list):
  ''' 
  Expands the input data dictionaries so that they each have the same 
  context (i.e. stations and observation times).
  '''
  # check for consisten units
  time_exp = data_list[0]['time_exponent']
  if not all(time_exp == d['time_exponent'] for d in data_list):
    raise ValueError('Data sets do not have consistent units')

  space_exp = data_list[0]['space_exponent']
  if not all(space_exp == d['space_exponent'] for d in data_list):
    raise ValueError('Data sets do not have consistent units')
  
  all_ids = np.hstack([d['id'] for d in data_list])
  all_lons = np.hstack([d['longitude'] for d in data_list])
  all_lats = np.hstack([d['latitude'] for d in data_list])
  all_times = np.hstack([d['time'] for d in data_list])
  unique_ids,idx = np.unique(all_ids,return_index=True)
  unique_lons = all_lons[idx]
  unique_lats = all_lats[idx]
  unique_times = np.arange(all_times.min(),all_times.max()+1)
  Nt,Nx = unique_times.shape[0],unique_ids.shape[0]
  out_list = []
  # create LUTs
  time_dict = dict(zip(unique_times,range(Nt)))
  id_dict = dict(zip(unique_ids,range(Nx)))
  for d in data_list:
    p = {}
    p['time_exponent'] = d['time_exponent']
    p['space_exponent'] = d['space_exponent']
    p['time'] = unique_times
    p['id'] = unique_ids 
    p['longitude'] = unique_lons
    p['latitude'] = unique_lats
    # find the indices that map the times and stations from d onto the 
    # unique times and stations
    tidx = [time_dict[i] for i in d['time']]
    sidx = [id_dict[i] for i in d['id']]
    for dir in ['east','north','vertical']:    
      p[dir] = np.empty((Nt,Nx))
      p[dir][...] = np.nan
      p[dir][np.ix_(tidx,sidx)] = d[dir]
      p[dir + '_std_dev'] = np.empty((Nt,Nx))
      p[dir + '_std_dev'][...] = np.inf
      p[dir + '_std_dev'][np.ix_(tidx,sidx)] = d[dir + '_std_dev']
      
    out_list += [p]

  return out_list


def _unit_string(space_exponent,time_exponent):
  ''' 
  returns a string indicating the units
  '''
  if space_exponent == 0:
    # if the space exponent is 0 then use units of microstrain
    space_str = '$\mathregular{\mu}$strain'
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
  # if the space exponent is 0 then use units of microstrain
  if space_exponent == 0:
    return 1.0e6 * (1.0/365.25)**time_exponent
  else:  
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
  scale_lon,scale_lat = bm(*ax.transData.inverted().transform(ax.transAxes.transform([0.15,0.1])),
                           inverse=True)
  scale_size = one_sigfig((bm.urcrnrx - bm.llcrnrx)/5.0)/1000.0
  bm.drawmapscale(scale_lon,scale_lat,scale_lon,scale_lat,scale_size,
                  ax=ax,barstyle='fancy',fontsize=10)
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
  logger.info('Viewing vector data sets ...')
  data_list = _common_context(data_list)

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
  su = [conv*d['east_std_dev'] for d in data_list]
  sv = [conv*d['north_std_dev'] for d in data_list]
  sz = [conv*d['vertical_std_dev'] for d in data_list]
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
  logger.info('Viewing strain data ...')
  data_dx,data_dy = _common_context([data_dx,data_dy])
  
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
  sxx = conv*data_dx['east_std_dev']
  syy = conv*data_dy['north_std_dev']
  sxy = 0.5*conv*np.sqrt(data_dx['north_std_dev']**2 + data_dy['east_std_dev']**2)

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',
                              facecolor='white')
  _setup_ts_ax(ts_ax)
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(lon,lat,resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = make_space_vert_smp(break_lons,break_lats,break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  interactive_strain_viewer(
    t,pos,exx,eyy,exy,sxx=sxx,syy=syy,sxy=sxy,
    map_ax=map_ax,ts_ax=ts_ax,time_labels=dates,
    station_labels=id,units=units,**kwargs)

  return
