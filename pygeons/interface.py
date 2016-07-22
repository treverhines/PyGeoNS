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

  time : (Nt,) array      # observation time in decimal years
  longitude : (Nx,) array
  latitude : (Nx,) array
  id : (Nx,) array        # station ID
  east : (Nt,Nx) array
  north : (Nt,Nx) array
  vertical : (Nt,Nx) array
  east_std : (Nt,Nx) array
  north_std : (Nt,Nx) array
  vertical_std : (Nt,Nx) array

'''
from __future__ import division
from functools import wraps
import pygeons.smooth
import pygeons.diff
import pygeons.view
import pygeons.clean
import pygeons.downsample
from pygeons.decyear import decyear_range
from pygeons.decyear import decyear_inv
from pygeons.decyear import decyear
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import logging
logger = logging.getLogger(__name__)


def _make_time_cuts(cut_dates):
  times = []
  if cut_dates is not None:
    # subtract half a day to ensure that the jump is observed at the 
    # indicated date
    times += [decyear(d,'%Y-%m-%d') - 0.5/365.25 for d in cut_dates]   

  out = pygeons.cuts.TimeCuts(times)
  return out
  

def _make_space_cuts(end1_lons,end1_lats,end2_lons,end2_lats,bm):
  if end1_lons is None: end1_lons = []
  if end1_lats is None: end1_lats = []
  if end2_lons is None: end2_lons = []
  if end2_lats is None: end2_lats = []
  end1 = [bm(i,j) for i,j in zip(end1_lons,end1_lats)]
  end2 = [bm(i,j) for i,j in zip(end2_lons,end2_lats)]
  out = pygeons.cuts.SpaceCuts(end1,end2)
  return out


def _check_data(data):
  ''' 
  make sure that the data set has all the necessary components and the 
  components have the right sizes
  '''
  # verify all components are here
  keys = ['time','id','longitude','latitude',
          'east','north','vertical',
          'east_std','north_std','vertical_std']
  for k in keys:
    if k not in data:
      raise ValueError('data set is missing "%s"' % k)
            
  Nt = len(data['time'])
  Nx = len(data['id'])
  # verify consistent station metadata
  if (len(data['longitude']) != Nx) | (len(data['latitude']) != Nx):
    raise ValueError('"longitude", "latitude", and "id" have inconsistent lengths')
    
  # verify data all has shape (Nt,Nx)
  data_keys = ['east','north','vertical','east_std','north_std','vertical_std']
  for d in data_keys: 
    if data[d].shape != (Nt,Nx): 
      raise ValueError('"%s" has shape %s but was expecting %s' % (d,data[d].shape,(Nt,Nx)))
     
  # verify that all the uncertainties have infs at the same component
  if np.any(np.isinf(data['east_std']) != np.isinf(data['north_std'])):
    raise ValueError('data with infinite uncertainty must have infinite uncertainty in all three directions') 

  if np.any(np.isinf(data['east_std']) != np.isinf(data['vertical_std'])):
    raise ValueError('data with infinite uncertainty must have infinite uncertainty in all three directions') 
      
  sigma_keys = ['east_std','north_std','vertical_std']
  for s in sigma_keys:
    if np.any(data[s] < 0.0):  
      raise ValueError('found negative uncertainties in "%s"' % s) 

  return   
  

def _check_io(fin):
  ''' 
  checks the input and output for functions that take and return data dictionaries
  '''
  @wraps(fin)
  def fout(data,*args,**kwargs):
    _check_data(data)
    data_out = fin(data,*args,**kwargs)
    _check_data(data_out)
    return data_out

  return fout  

def _log_call(fin):
  ''' 
  Notifies the user of calls to a function and prints all but the 
  first positional input argument, which is the data dictionary
  '''
  @wraps(fin)
  def fout(data,*args,**kwargs):
    str = 'calling function %s :\n' % fin.__name__
    str += '    positional arguments following data :\n'
    for a in args:
      str += '        %s\n' % a
    str += '    key word arguments :\n'
    for k,v in kwargs.iteritems():
      str += '        %s : %s\n' % (k,v)

    logger.debug(str)
    return fin(data,*args,**kwargs)

  return fout
      

def _check_compatibility(data_list):
  ''' 
  make sure that each data set contains the same stations and times
  '''
  for d in data_list:
    _check_data(d)
    
  # compare agains the first data set
  time = data_list[0]['time']  
  id = data_list[0]['id']  
  lon = data_list[0]['longitude']  
  lat = data_list[0]['latitude']  
  for d in data_list[1:]:
    if len(time) != len(d['time']):
      raise ValueError('data sets have inconsistent number of time epochs')
    if len(id) != len(d['id']):
      raise ValueError('data sets have inconsistent number of stations')
    if len(lon) != len(d['longitude']):
      raise ValueError('data sets have inconsistent number of stations')
    if len(lat) != len(d['latitude']):
      raise ValueError('data sets have inconsistent number of stations')

    # this makes sure that times are within ~12 hours of eachother
    if not np.all(np.isclose(time,d['time'],atol=1e-3)):
      raise ValueError('data sets do not have the same times epochs')

    # make sure the stations have the same names
    if not np.all(id==d['id']):
      raise ValueError('data sets do not have the same stations')

    # this makes sure the stations are within a few hundred meters of eachother
    if not np.all(np.isclose(lon,d['longitude'],atol=1e-3)):
      raise ValueError('data sets do not have the same station positions')
    if not np.all(np.isclose(lat,d['latitude'],atol=1e-3)): 
      raise ValueError('data sets do not have the same station positions')
    
  return  


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
    str = 'x : %g m  y : %g m  ' % (x,y)
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
                     

def _setup_ts_ax(ax_lst):
  # display time in decyear and date on time series plot
  def ts_coord_string(x,y):                         
    str = 'time : %g  ' % x
    str += '(date : %s)' % decyear_inv(x,'%Y-%m-%d')
    return str 

  ax_lst[0].format_coord = ts_coord_string
  ax_lst[1].format_coord = ts_coord_string
  ax_lst[2].format_coord = ts_coord_string
  return


@_check_io
@_log_call
def smooth_space(data,length_scale=None,fill=False,
                 cut_endpoint1_lons=None,cut_endpoint1_lats=None,
                 cut_endpoint2_lons=None,cut_endpoint2_lats=None):
  ''' 
  Spatially smooths the data set. Data is treated as a stochastic 
  variable where its Laplacian is modeled as white noise.
  
  Parameters
  ----------
    data : dict
      data dictionary
      
    length_scale : float, optional
      length scale of the smoothed data. Defaults to 10X the average 
      shortest distance between stations. This is specified in meters.
       
    fill : bool, optional
      whether to make an estimate at masked data. Filling masked data 
      can make this function slower and more likely to encounter a 
      singular matrix. Defaults to False.
    
    cut_endpoints{1,2}_{lons,lat} :  lst, optional
      coordinates of the spatial discontinuty line segments. 
      Smoothness is not enforced across these discontinuities
  
  Returns
  -------
    out : dict
      output data dictionary
      
  '''
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  
  space_cuts = _make_space_cuts(cut_endpoint1_lons,cut_endpoint1_lats,
                                cut_endpoint2_lons,cut_endpoint2_lats,bm)
  ds = pygeons.diff.disp_laplacian()
  ds['space']['cuts'] = space_cuts
  ds = [ds]

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    u_smooth = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,
                                     diff_specs=ds,
                                     time_scale=0.0,
                                     length_scale=length_scale,
                                     fill=fill)
    sigma_smooth = np.zeros(u.shape)
    if not fill:
      sigma_smooth[np.isinf(sigma)] = np.inf

    out[dir] = u_smooth
    out[dir+'_std'] = sigma_smooth

  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  return out


@_check_io
@_log_call
def smooth_time(data,time_scale=None,fill=False,
                cut_dates=None):
  ''' 
  Temporally smooths the data set. The data is modeled as a stochastic 
  variables where its second time derivative is white noise (i.e. 
  integrated Brownain motion).  
  
  Parameters
  ----------
    data : dict
      data dictionary
      
    time_scale : float, optional
      time scale of the smoothed data. Defaults to 10X the time sample 
      period. This is specified in years

    fill : bool, optional
      whether to make an estimate at masked data. Filling masked data 
      can make this function slower and more likely to encounter a 
      singular matrix. Defaults to False.
    
    cut_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed 
    
  Returns
  -------
    out : dict
      output data dictionary

  '''
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  time_cuts = _make_time_cuts(cut_dates)
  ds = pygeons.diff.acc()
  ds['time']['cuts'] = time_cuts
  ds = [ds]

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    u_smooth = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,
                                     diff_specs=ds,
                                     time_scale=time_scale,
                                     length_scale=0.0,
                                     fill=fill)
    sigma_smooth = np.zeros(u.shape)
    if not fill:
      sigma_smooth[np.isinf(sigma)] = np.inf

    out[dir] = u_smooth
    out[dir+'_std'] = sigma_smooth

  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']

  return out
  

@_check_io
@_log_call
def smooth(data,time_scale=None,length_scale=None,fill=False,
           cut_endpoint1_lons=None,cut_endpoint1_lats=None,
           cut_endpoint2_lons=None,cut_endpoint2_lats=None,
           cut_dates=None):
  ''' 
  Spatially and temporally smooths the data set. Data is treated as a 
  stochastic variable where its second time derivative is white noise 
  and its Laplacian is white noise.
  
  Parameters
  ----------
    data : dict
      data dictionary
      
    time_scale : float, optional
      Time scale of the smoothed data. Defaults to 10X the time sample 
      period. This is specified in years

    length_scale : float, optional
      Length scale of the smoothed data. Defaults to 10X the average 
      shortest distance between stations. This is specified in meters
       
    fill : bool, optional
      Whether to make an estimate at masked data. Filling masked data 
      can make this function slower and more likely to encounter a 
      singular matrix. Defaults to False.
    
    cut_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed 
    
  Returns
  -------
    out : dict
      output data dictionary
      
  '''
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  space_cuts = _make_space_cuts(cut_endpoint1_lons,cut_endpoint1_lats,
                                cut_endpoint2_lons,cut_endpoint2_lats,bm)
  time_cuts = _make_time_cuts(cut_dates)
  ds1 = pygeons.diff.acc()  
  ds1['time']['cuts'] = time_cuts
  ds2 = pygeons.diff.disp_laplacian()
  ds2['space']['cuts'] = space_cuts
  ds = [ds1,ds2]

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    u_smooth = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,
                                     diff_specs=ds,
                                     time_scale=time_scale,
                                     length_scale=length_scale,
                                     fill=fill)
    sigma_smooth = np.zeros(u.shape)
    if not fill:
      sigma_smooth[np.isinf(sigma)] = np.inf

    out[dir] = u_smooth
    out[dir+'_std'] = sigma_smooth

  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']

  return out
           

@_check_io
@_log_call
def diff(data,dt=0,dx=0,dy=0,
         cut_endpoint1_lons=None,cut_endpoint1_lats=None,
         cut_endpoint2_lons=None,cut_endpoint2_lats=None,
         cut_dates=None):
  ''' 
  Calculates a mixed partial derivative of the data set. The spatial 
  coordinates are in units of meters and time is in years. The output 
  then has units of {input units} per meter**(dx+dy) per year**dt, 
  where dx, dy, and dt are the derivative orders described below.
  
  Parameters
  ----------
    data : dict
      data dictionary
    
    dt : int, optional
      time derivative order 

    dx : int, optional
      x derivative order. The x direction depends on the map 
      projection but is roughly east.

    dy : int, optional
      y derivative order. The y direction depends on the map 
      projection but is roughly north
      
    cut_endpoints{1,2}_{lons,lat} :  lst, optional
      coordinates of the spatial discontinuty line segments
  
    cut_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed
      
  Returns 
  -------
    out : dict
      output dictionary

  '''
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  space_cuts = _make_space_cuts(cut_endpoint1_lons,cut_endpoint1_lats,
                                cut_endpoint2_lons,cut_endpoint2_lats,bm)
  time_cuts = _make_time_cuts(cut_dates)
  # form DiffSpecs instance
  ds = pygeons.diff.DiffSpecs()
  ds['time']['diffs'] = [(dt,)]
  ds['time']['cuts'] = time_cuts
  ds['space']['diffs'] = [(dx,dy)]
  ds['space']['cuts'] = space_cuts

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    mask = np.isinf(data[dir+'_std'])
    u_diff = pygeons.diff.diff(data['time'],pos,u,ds,mask=mask)
    sigma_diff = np.zeros(u.shape)
    sigma_diff[mask] = np.inf
    out[dir] = u_diff
    out[dir+'_std'] = sigma_diff

  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  
  return out
  

@_check_io
@_log_call
def downsample(data,sample_period,start_date=None,stop_date=None,
               cut_dates=None):
  ''' 
  downsamples the data set 
  
  Parameters
  ----------
    data : dict
      data dictionary
      
    sample_period : int
      sample period of the output data set in days. Output data is 
      computed using a running mean with this width. This should be an 
      odd integer in order to avoid double counting the observations 
      at some days
      
    start_date : str, optional
      start date of output data set in YYYY-MM-DD. Uses the start date 
      of *data* if not provided

    stop_date : str, optional
      stop date of output data set in YYYY-MM-DD. Uses the stop date 
      of *data* if not provided

    cut_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed
      
  Returns
  -------
    out_dict : dict
      output data dictionary

  '''
  time_cuts = _make_time_cuts(cut_dates)
  
  # if the start and stop time are now specified then use the min and 
  # max times
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  if start_date is None:
    # make sure the start and end date are rounded the nearest start 
    # of the day
    start_date = decyear_inv(data['time'].min()+0.5/365.25,'%Y-%m-%d')
  if stop_date is None:
    stop_date = decyear_inv(data['time'].max()+0.5/365.25,'%Y-%m-%d')
  
  # make sure that the sample period is an integer multiple of days. 
  # This is needed to be able to write to csv files without any data 
  # loss.
  sample_period = int(sample_period)
  time_itp = decyear_range(start_date,stop_date,sample_period,'%Y-%m-%d')

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    u_ds,sigma_ds = pygeons.downsample.downsample(
                      data['time'],time_itp,pos,u,
                      sigma=sigma,time_cuts=time_cuts)
               
    out[dir] = u_ds
    out[dir+'_std'] = sigma_ds

  out['time'] = time_itp
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  return out


@_check_io
@_log_call
def zero(data,zero_date,radius,cut_dates=None):
  ''' 
  Estimates and removes the displacements at the indicated time. The 
  offset is calculated with a weighted mean centered about the 
  *zero_time*. *radius* specifies the temporal extent of data used in 
  the weighted mean
  
  Parameters
  ----------
    data : dict
    
    zero_date : str
      Zero displacements at this date which is specified in YYYY-MM-DD
      
    radius : int
      number of days before and after *zero_date* to use in 
      calculating the offset. 
        
    cut_dates : lst, optional
      list of time discontinuities in YYYY-MM-DD. This date should be 
      when the discontinuity is first observed

  Returns
  -------
    out : dict
      Output data dictionary

  '''
  zero_time = decyear(zero_date,'%Y-%m-%d')
  
  # add half a day to prevent any rounding errors and then convert to 
  # years
  radius = (int(radius) + 0.5)/365.25
  
  time_cuts = _make_time_cuts(cut_dates)
  vert,smp = time_cuts.get_vert_and_smp([0.0,0.0])
  
  out = {}
  for dir in ['east','north','vertical']:
    itp = pygeons.downsample.MeanInterpolant(
            data['time'][:,None],data[dir],
            sigma=data[dir +'_std'],
            vert=vert,smp=smp)    
    u_zero,sigma_zero = itp([[zero_time]],radius=radius) 
    out[dir] = data[dir] - u_zero
    out[dir + '_std'] = np.sqrt(data[dir + '_std']**2 + sigma_zero**2)
  
  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude']
  out['id'] = data['id']
  return out


@_check_io
@_log_call
def clean(data,resolution='i',
         cut_endpoint1_lons=None,cut_endpoint1_lats=None,
         cut_endpoint2_lons=None,cut_endpoint2_lats=None,
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
  _setup_ts_ax(ts_ax)

  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw cuts if there are any
  space_cuts = _make_space_cuts(cut_endpoint1_lons,cut_endpoint1_lats,
                                cut_endpoint2_lons,cut_endpoint2_lats,bm)
  vert,smp = space_cuts.get_vert_and_smp(0.0)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  t = data['time']
  u,v,z = data['east'],data['north'],data['vertical']
  su,sv,sz = data['east_std'],data['north_std'],data['vertical_std']
  clean_data = pygeons.clean.clean(
                 t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
                 converter=bm,map_ax=map_ax,ts_ax=ts_ax,
                 station_names=data['id'],**kwargs)

  out = {}
  out['time'] = data['time']
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  out['east'] = clean_data[0]          
  out['north'] = clean_data[1]          
  out['vertical'] = clean_data[2]          
  out['east_std'] = clean_data[3]   
  out['north_std'] = clean_data[4]          
  out['vertical_std'] = clean_data[5]   

  return out


@_log_call
def view(data_list,resolution='i',
         cut_endpoint1_lons=None,cut_endpoint1_lats=None,
         cut_endpoint2_lons=None,cut_endpoint2_lats=None,
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
  _check_compatibility(data_list)
  t = data_list[0]['time']
  lon = data_list[0]['longitude']
  lat = data_list[0]['latitude']
  id = data_list[0]['id']

  u = [d['east'] for d in data_list]
  v = [d['north'] for d in data_list]
  z = [d['vertical'] for d in data_list]
  su = [d['east_std'] for d in data_list]
  sv = [d['north_std'] for d in data_list]
  sz = [d['vertical_std'] for d in data_list]

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax)
   
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw cuts if there are any
  space_cuts = _make_space_cuts(cut_endpoint1_lons,cut_endpoint1_lats,
                                cut_endpoint2_lons,cut_endpoint2_lats,bm)
  vert,smp = space_cuts.get_vert_and_smp(0.0)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.view.view(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    ts_ax=ts_ax,map_ax=map_ax,station_names=id,**kwargs)

  return
