''' 
functions which are called by the PyGeoNS executables. 

These wrappers are designed to have input arguments that can be easily 
specified through the command line. This limits the full functionality 
of the functions being called in the interest of usability. 

These functions are thin wrappers which do conversions from geodetic 
to cartesian coordinate systems and make some sensible simplifying 
assumptions about the input arguments

every function takes a data dictionary as input. The data dictionary 
contains the following items

  time : (Nt,) array
  longitude : (Nx,) array
  latitude : (Nx,) array
  id : (Nx, array)
  east : (Nt,Nx) array
  north : (Nt,Nx) array
  vertical : (Nt,Nx) array
  east_std : (Nt,Nx) array
  north_std : (Nt,Nx) array
  vertical_std : (Nt,Nx) array

'''
from __future__ import division
import pygeons.smooth
import pygeons.diff
import pygeons.view
import pygeons.clean
import pygeons.downsample
from pygeons.decyear import decyear_range
from pygeons.decyear import decyear_inv
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

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
  

def _check_compatibility(data_list):
  ''' 
  make sure that each data set contains the same stations and times
  '''
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
    if not np.all(np.isclose(time,d['time'],atol=1e-3)):
      raise ValueError('data sets do not have the same times epochs')
    if not np.all(id==d['id']):
      raise ValueError('data sets do not have the same stations')
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


def smooth_space(data,length_scale=None,fill=False,
                 cut_endpoint1_lons=None,
                 cut_endpoint1_lats=None,
                 cut_endpoint2_lons=None,
                 cut_endpoint2_lats=None):

  if cut_endpoint1_lons is None: cut_endpoint1_lons = []
  if cut_endpoint1_lats is None: cut_endpoint1_lats = []
  if cut_endpoint2_lons is None: cut_endpoint2_lons = []
  if cut_endpoint2_lats is None: cut_endpoint2_lats = []

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  
  cut_endpoints1 = [bm(*i) for i,j in zip(cut_endpoint1_lons,cut_endpoint1_lats)]
  cut_endpoints2 = [bm(*i) for i,j in zip(cut_endpoint2_lons,cut_endpoint2_lats)]
  ds = pygeons.diff.disp_laplacian()
  ds['space']['cuts'] = pygeons.cuts.SpaceCuts(cut_endpoints1,cut_endpoints2)
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


def smooth_time(data,time_scale=None,fill=False,
                cut_times=None):

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  ds = pygeons.diff.acc()
  ds['time']['cuts'] = pygeons.cuts.TimeCuts(cut_times)
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
  

def smooth(data,time_scale=None,length_scale=None,fill=False,
           cut_endpoint1_lons=None,
           cut_endpoint1_lats=None,
           cut_endpoint2_lons=None,
           cut_endpoint2_lats=None,
           cut_times=None):

  if cut_endpoint1_lons is None: cut_endpoint1_lons = []
  if cut_endpoint1_lats is None: cut_endpoint1_lats = []
  if cut_endpoint2_lons is None: cut_endpoint2_lons = []
  if cut_endpoint2_lats is None: cut_endpoint2_lats = []

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  cut_endpoints1 = [bm(*i) for i,j in zip(cut_endpoint1_lons,cut_endpoint1_lats)]
  cut_endpoints2 = [bm(*i) for i,j in zip(cut_endpoint2_lons,cut_endpoint2_lats)]
  ds1 = pygeons.diff.acc()  
  ds1['time']['cuts'] = pygeons.cuts.TimeCuts(cut_times)
  ds2 = pygeons.diff.disp_laplacian()
  ds2['space']['cuts'] = pygeons.cuts.SpaceCuts(cut_endpoints1,cut_endpoints2)
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
           

def diff(data,dt=0,dx=0,dy=0,
         cut_endpoint1_lons=None,
         cut_endpoint1_lats=None,
         cut_endpoint2_lons=None,
         cut_endpoint2_lats=None,
         cut_times=None):

  if cut_endpoint1_lons is None: cut_endpoint1_lons = []
  if cut_endpoint1_lats is None: cut_endpoint1_lats = []
  if cut_endpoint2_lons is None: cut_endpoint2_lons = []
  if cut_endpoint2_lats is None: cut_endpoint2_lats = []

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  cut_endpoints1 = [bm(*i) for i,j in zip(cut_endpoint1_lons,cut_endpoint1_lats)]
  cut_endpoints2 = [bm(*i) for i,j in zip(cut_endpoint2_lons,cut_endpoint2_lats)]

  # form DiffSpecs instance
  ds = pygeons.diff.DiffSpecs()
  ds['time']['diffs'] = [(dt,)]
  ds['time']['cuts'] = pygeons.cuts.TimeCuts(cut_times)
  ds['space']['diffs'] = [(dx,dy)]
  ds['space']['cuts'] = pygeons.cuts.SpaceCuts(cut_endpoints1,cut_endpoints2)

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
  

def downsample(data,period,start=None,stop=None,cut_times=None):
  # if the start and stop time are now specified then use the min and 
  # max times
  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  if start is None:
    start = decyear_inv(data['time'].min()+0.5/365.25,'%Y-%m-%d')
  if stop is None:
    stop = decyear_inv(data['time'].max()+0.5/365.25,'%Y-%m-%d')
  
  # make sure that the sample period is an integer multiple of days. 
  # This is needed to be able to write to csv files without any data 
  # loss.
  period = int(period)
  
  time_itp = decyear_range(start,stop,period,'%Y-%m-%d')

  out = {}
  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    u_ds,sigma_ds = pygeons.downsample.downsample(
                      data['time'],time_itp,
                      pos,u,sigma,cut_times)
               
    out[dir] = u_ds
    out[dir+'_std'] = sigma_ds

  out['time'] = time_itp
  out['longitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  return out

def zero(data,**kwargs):
  return

def clean(data,resolution='i',**kwargs):
  fig,ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution)
  bm.drawcountries()
  bm.drawstates() 
  bm.drawcoastlines()
  mer,par =  _get_meridians_and_parallels(bm,3)
  bm.drawmeridians(mer,
                   labels=[0,0,0,1],dashes=[2,2],
                   ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))
  bm.drawparallels(par,
                   labels=[1,0,0,0],dashes=[2,2],
                   ax=ax,zorder=1,color=(0.3,0.3,0.3,1.0))

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  t = data['time']
  u,v,z = data['east'],data['north'],data['vertical']
  su,sv,sz = data['east_std'],data['north_std'],data['vertical_std']
  clean_data = pygeons.clean.clean(
                 t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
                 converter=bm,map_ax=ax,**kwargs)

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


def view(data_list,resolution='i',**kwargs):
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

  fig,ax = plt.subplots(num='Map View',facecolor='white')
  bm = _make_basemap(lon,lat,
                     resolution=resolution)
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
  
  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.view.view(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    map_ax=ax,station_names=id,**kwargs)

  return
