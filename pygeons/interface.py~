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
  east_pert : (Np,Nt,Nx) array
  north_pert : (Np,Nt,Nx) array
  vertical_pert : (Np,Nt,Nx) array

'''
from __future__ import division
from functools import wraps
import rbf.geometry
import pygeons.smooth
import pygeons.diff
import pygeons.view
import pygeons.clean
import pygeons.downsample
from pygeons.downsample import weighted_mean
from pygeons.dateconv import decday_inv
from pygeons.dateconv import decday
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import logging
import copy
logger = logging.getLogger(__name__)


def _make_time_cuts(cut_dates):
  out = []
  if cut_dates is not None:
    # subtract half a day to get rid of any ambiguity about what day 
    # the dislocation is observed
    out += [decday(d,'%Y-%m-%d') - 0.5 for d in cut_dates]   

  return np.array(out)
  

def _make_space_cuts(end1_lons,end1_lats,end2_lons,end2_lats,bm):
  if end1_lons is None: end1_lons = []
  if end1_lats is None: end1_lats = []
  if end2_lons is None: end2_lons = []
  if end2_lats is None: end2_lats = []
  end1 = [bm(i,j) for i,j in zip(end1_lons,end1_lats)]
  end2 = [bm(i,j) for i,j in zip(end2_lons,end2_lats)]
  end1 = np.array(end1).reshape((-1,2))
  end2 = np.array(end2).reshape((-1,2))
  out = np.concatenate((end1[:,None,:],end2[:,None,:]),axis=1)
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
     
  pert_keys = ['east_pert','north_pert','vertical_pert']
  for p in pert_keys:
    if p in data:
      if data[p].shape[1:] != (Nt,Nx): 
        raise ValueError('"%s" has shape %s but was expecting (...,%s,%s)' % (d,data[p].shape,Nt,Nx))

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
  

def _check_compatibility(data_list):
  ''' 
  make sure that each data set contains the same stations and times
  '''
  for d in data_list:
    _check_data(d)
    
  # compare agains the first data set
  time = data_list[0]['time']  
  id = data_list[0]['id']  
  for d in data_list[1:]:
    if len(time) != len(d['time']):
      raise ValueError('data sets have inconsistent number of time epochs')
    if len(id) != len(d['id']):
      raise ValueError('data sets have inconsistent number of stations')

    # make sure each data set has the same times
    if not np.all(time==d['time']):
      raise ValueError('data sets do not have the same times epochs')

    # make sure each data set has the same stations
    if not np.all(id==d['id']):
      raise ValueError('data sets do not have the same stations')

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
      

def _perturbation_uncertainty(fin):
  ''' 
  set sigma equal to the standard deviation of the perturbations
  '''
  @wraps(fin)
  def fout(data,*args,**kwargs):
    out = fin(data,*args,**kwargs)
    for dir in ['east','north','vertical']:
      # only use the perturbations for uncertainty if there are more 
      # than zero
      if out[dir+'_pert'].shape[0] > 0:
        mask = np.isinf(out[dir+'_std'])
        sigma = np.std(out[dir+'_pert'],axis=0)
        sigma[mask] = np.inf
        out[dir+'_std'] = sigma

    return out

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
@_log_call
@_perturbation_uncertainty
def tsmooth(data,time_scale=None,fill='none',cut_dates=None):
  ''' 
  time smoothing
  '''
  kind = "time"
  return _smooth(data,time_scale=time_scale,
                 length_scale=0.0,
                 kind=kind,fill=fill,
                 cut_dates=cut_dates)
                 
@_check_io
@_log_call
@_perturbation_uncertainty
def ssmooth(data,length_scale=None,
            stencil_size=None,fill='none',
            cut_lons1=None,cut_lats1=None,
            cut_lons2=None,cut_lats2=None):
  ''' 
  space smoothing
  '''
  kind = "space"
  return _smooth(data,time_scale=0.0,
                 length_scale=length_scale,
                 kind=kind,fill=fill,
                 stencil_size=stencil_size,
                 cut_lons1=cut_lons1,cut_lats1=cut_lats1,
                 cut_lons2=cut_lons2,cut_lats2=cut_lats2)


def _smooth(data,time_scale=None,length_scale=None,
           kind='both',fill='none',
           stencil_size=None,
           cut_lons1=None,cut_lats1=None,
           cut_lons2=None,cut_lats2=None,
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
      period. This is specified in days

    length_scale : float, optional
      Length scale of the smoothed data. Defaults to 10X the average 
      shortest distance between stations. This is specified in meters
       
    kind : str, optional
      either "time" or "space"
      
    fill : str, optional
      either "none", "interpolate", or "extrapolate". Indicates when 
      and where to make a smoothed estimate. "none" : output only 
      where data is not missing. "interpolate" : output where data is 
      not missing and where time interpolation is possible. "all" : 
      output at all stations and times.
    
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

  space_cuts = _make_space_cuts(cut_lons1,cut_lats1,
                                cut_lons2,cut_lats2,bm)
  time_cuts = _make_time_cuts(cut_dates)
  
  ds1 = pygeons.diff.acc()  
  ds1['time']['cuts'] = time_cuts
  ds2 = pygeons.diff.disp_laplacian()
  ds2['space']['cuts'] = space_cuts
  ds2['space']['stencil_size'] = stencil_size
  if kind == 'time':
    ds = [ds1]  

  elif kind == 'space':
    ds = [ds2]
      
  elif kind == 'both':
    ds = [ds1,ds2]
  
  else:
    raise ValueError('*kind* must be "time", "space", or "both"')  

  out = copy.deepcopy(data)
  for dir in ['east','north','vertical']:
    u = data[dir]
    p = data[dir+'_pert']
    sigma = data[dir+'_std']
      
    up = np.concatenate((u[None,:,:],p),axis=0)
    up_smooth = pygeons.smooth.smooth(
                  data['time'],pos,up,
                  sigma=sigma,
                  diff_specs=ds,
                  time_scale=time_scale,
                  length_scale=length_scale,
                  fill=fill)
    # if the returned value is np.nan, then it is masked. Make sure 
    # that the corresponding sigma is np.inf
    mask = np.isnan(up_smooth[0,:,:])
    sigma_smooth = np.zeros(sigma.shape)
    sigma_smooth[mask] = np.inf
    u_smooth = up_smooth[0,:,:]
    p_smooth = up_smooth[1:,:,:]

    out[dir] = u_smooth
    out[dir+'_pert'] = p_smooth
    out[dir+'_std'] = sigma_smooth
    
  return out
           

@_check_io
@_log_call
@_perturbation_uncertainty
def tdiff(data,dt,cut_dates=None):
  ''' 
  time differentiation
  '''
  return _diff(data,dt=dt,cut_dates=cut_dates)   


@_check_io
@_log_call
@_perturbation_uncertainty
def sdiff(data,dx,dy,stencil_size=None,
          cut_lons1=None,cut_lats1=None,
          cut_lons2=None,cut_lats2=None):
  ''' 
  space differentiation
  '''
  return _diff(data,dx=dx,dy=dy,stencil_size=stencil_size,
               cut_lons1=cut_lons1,cut_lats1=cut_lats1,
               cut_lons2=cut_lons2,cut_lats2=cut_lats2)


def _diff(data,dt=None,dx=None,dy=None,
          stencil_size=None,
          cut_lons1=None,cut_lats1=None,
          cut_lons2=None,cut_lats2=None,
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

  space_cuts = _make_space_cuts(cut_lons1,cut_lats1,
                                cut_lons2,cut_lats2,bm)
  time_cuts = _make_time_cuts(cut_dates)
  # form DiffSpecs instance
  ds = pygeons.diff.DiffSpecs()
  if dt is not None:
    ds['time']['diffs'] = [(dt,)]
    ds['time']['cuts'] = time_cuts

  if (dx is not None) | (dy is not None):     
    if dx is None: dx = 0
    if dy is None: dy = 0
    ds['space']['diffs'] = [(dx,dy)]
    ds['space']['cuts'] = space_cuts
    ds['space']['stencil_size'] = stencil_size

  out = copy.deepcopy(data)
  for dir in ['east','north','vertical']:
    u = data[dir]
    p = data[dir+'_pert']
    mask = np.isinf(data[dir+'_std'])

    up = np.concatenate((u[None,:,:],p),axis=0)
    up_diff = pygeons.diff.diff(data['time'],pos,up,ds,mask=mask)

    u_diff = up_diff[0,:,:]
    p_diff = up_diff[1:,:,:]
    sigma_diff = np.zeros(u.shape)
    sigma_diff[mask] = np.inf
    out[dir] = u_diff
    out[dir+'_pert'] = p_diff
    out[dir+'_std'] = sigma_diff

  return out
  

@_check_io
@_log_call
@_perturbation_uncertainty
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
    start_date = decday_inv(data['time'].min(),'%Y-%m-%d')
  if stop_date is None:
    stop_date = decday_inv(data['time'].max(),'%Y-%m-%d')
  
  start_time = int(decday(start_date,'%Y-%m-%d'))
  stop_time = int(decday(stop_date,'%Y-%m-%d'))
  sample_period = int(sample_period)
  time_itp = np.arange(start_time,stop_time+1,sample_period)

  out = copy.deepcopy(data)
  out['time'] = time_itp
  for dir in ['east','north','vertical']:
    u = data[dir]
    p = data[dir+'_pert']
    up = np.concatenate((u[None,:,:],p),axis=0)
    sigma = data[dir+'_std']
    up_ds,sigma_ds = pygeons.downsample.downsample(
                       data['time'],time_itp,pos,up,
                       sigma=sigma,time_cuts=time_cuts)
                       
    u_ds = up_ds[0,:,:]
    p_ds = up_ds[1:,:,:]
               
    out[dir] = u_ds
    out[dir+'_pert'] = p_ds
    out[dir+'_std'] = sigma_ds

  return out


@_check_io
@_log_call
@_perturbation_uncertainty
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
  radius = int(radius)
  zero_time = int(decday(zero_date,'%Y-%m-%d'))
  time_cuts = _make_time_cuts(cut_dates)
  
  vert = np.array(time_cuts).reshape((-1,1))
  smp = np.arange(vert.shape[0]).reshape((-1,1))
  
  out = copy.deepcopy(data)
  for dir in ['east','north','vertical']:
    u = data[dir]
    p = data[dir + '_pert']
    sigma = data[dir+'_std']

    up = np.concatenate((u[None,:,:],p),axis=0)
    # expand sigma to the size of up
    sigma_ext = sigma[None,:,:].repeat(up.shape[0],axis=0)
    
    tidx, = np.nonzero((data['time'] >= (zero_time - radius)) &
                       (data['time'] <= (zero_time + radius)))

    # make sure that there are no intersections
    time = data['time'][tidx][:,None]
    zero_ext = np.repeat(zero_time,len(tidx))[:,None]
    cross = rbf.geometry.intersection_count(time,zero_ext,vert,smp)
    # remove any indices that crossed a boundary
    tidx = tidx[cross==0]
    
    offset,sigma_offset = weighted_mean(up[:,tidx,:],sigma_ext[:,tidx,:],axis=1)
    # just take the first one
    sigma_offset = sigma_offset[0,:]
    up_zero = up - offset[:,None,:]
    u_zero = up_zero[0,:,:]
    p_zero = up_zero[1:,:,:]
    sigma_zero = np.sqrt(sigma**2 + sigma_offset[None,:]**2)
           
    out[dir] = u_zero
    out[dir+'_pert'] = p_zero
    out[dir + '_std'] = sigma_zero
  
  return out


@_check_io
@_log_call
@_perturbation_uncertainty
def perturb(data,N):
  ''' 
  adds a displacement perturbations to the data dictionary
  
  Parameters
  ----------
    data : dict
    
    N : int
      number of perturbations

  Returns
  -------
    out : dict
      data dict with *pert* entries

  '''
  out = copy.deepcopy(data)
  for dir in ['east','north','vertical']:
    sigma = data[dir+'_std']
    sigma = sigma[None,:,:].repeat(N,axis=0)
    # dont add noise for data where sigma is inf or 0.0
    is_valid = (sigma > 0.0) & ~np.isinf(sigma)
    noise = np.zeros(sigma.shape)
    noise[is_valid] = np.random.normal(0.0,sigma[is_valid])
    out[dir+'_pert'] = data[dir] + noise

  return out


@_check_io
@_log_call
@_perturbation_uncertainty
def clean(data,resolution='i',
          cut_lons1=None,cut_lats1=None,
          cut_lons2=None,cut_lats2=None,
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
    
  Note
  ----
    Currently, if clean has been called after perturb, then the 
    uncertainties resulting from removing jumps will not be added to 
    the output. This is because during cleaning, the perturbations are 
    not touched at all and their variances will remain the same. Newly 
    masked data (due to removing outliers or jumps) will be masked in 
    the output
    
  '''
  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax,data['time'])

  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw cuts if there are any
  space_cuts = _make_space_cuts(cut_lons1,cut_lats1,
                                cut_lons2,cut_lats2,bm)

  vert = np.array(space_cuts).reshape((-1,2))
  smp = np.arange(vert.shape[0]).reshape((-1,2))
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  t = data['time']
  dates = [decday_inv(ti,'%Y-%m-%d') for ti in t]
  u,v,z = data['east'],data['north'],data['vertical']
  up,vp,zp = data['east_pert'],data['north_pert'],data['vertical_pert']
  up = np.concatenate((u[None,:,:],up),axis=0)
  vp = np.concatenate((v[None,:,:],vp),axis=0)
  zp = np.concatenate((z[None,:,:],zp),axis=0)
  su,sv,sz = data['east_std'],data['north_std'],data['vertical_std']
  clean_data = pygeons.clean.clean(
                 t,pos,u=up,v=vp,z=zp,su=su,sv=sv,sz=sz,
                 map_ax=map_ax,ts_ax=ts_ax,
                 time_labels=dates,
                 station_labels=data['id'],
                 **kwargs)

  out = copy.deepcopy(data)
  out['east'] = clean_data[0][0,:,:]
  out['north'] = clean_data[1][0,:,:]        
  out['vertical'] = clean_data[2][0,:,:]          
  out['east_pert'] = clean_data[0][1:,:,:]
  out['north_pert'] = clean_data[1][1:,:,:]        
  out['vertical_pert'] = clean_data[2][1:,:,:]          
  out['east_std'] = clean_data[3]   
  out['north_std'] = clean_data[4]          
  out['vertical_std'] = clean_data[5]   

  return out


@_log_call
def view(data_list,resolution='i',
         cut_lons1=None,cut_lats1=None,
         cut_lons2=None,cut_lats2=None,
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
  dates = [decday_inv(ti,'%Y-%m-%d') for ti in t]

  u = [d['east'] for d in data_list]
  v = [d['north'] for d in data_list]
  z = [d['vertical'] for d in data_list]
  su = [d['east_std'] for d in data_list]
  sv = [d['north_std'] for d in data_list]
  sz = [d['vertical_std'] for d in data_list]

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax,data_list[0]['time'])
   
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,
                     resolution=resolution)
  _setup_map_ax(bm,map_ax)

  # draw cuts if there are any
  space_cuts = _make_space_cuts(cut_lons1,cut_lats1,
                                cut_lons2,cut_lats2,bm)

  vert = np.array(space_cuts).reshape((-1,2))
  smp = np.arange(vert.shape[0]).reshape((-1,2))
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'r-',lw=2,zorder=2)

  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.view.view(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    ts_ax=ts_ax,map_ax=map_ax,
    station_labels=id,
    time_labels=dates,
    **kwargs)

  return
