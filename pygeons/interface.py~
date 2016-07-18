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
import pygeons.smooth
import pygeons.diff
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


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

  out = {}
  out['time'] = data['time']
  out['logitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']


  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  
  cut_endpoints1 = [bm(*i) for i,j in zip(cut_endpoint1_lons,cut_endpoint1_lats)]
  cut_endpoints2 = [bm(*i) for i,j in zip(cut_endpoint2_lons,cut_endpoint2_lats)]
  ds = pygeons.diff.disp_laplacian()
  ds['space']['cuts'] = pygeons.cuts.SpaceCuts(cut_endpoints1,cut_endpoints2)
  ds = [ds]

  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    out[dir] = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,diff_specs=ds,
                                     length_scale=length_scale,
                                     time_scale=0.0,
                                     fill=fill)
    out[dir+'_std'] = np.zeros(u.shape)    

  return out


def smooth_time(data,time_scale=None,fill=False,
                cut_times=None):

  if cut_times is None: cut_times = []

  out = {}
  out['time'] = data['time']
  out['logitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  ds = pygeons.diff.acc()
  ds['time']['cuts'] = pygeons.cuts.TimeCuts(cut_times)
  ds = [ds]

  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    out[dir] = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,diff_specs=ds,
                                     length_scale=0.0,
                                     time_scale=time_scale,
                                     fill=fill)
    out[dir+'_std'] = np.zeros(u.shape)    

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
  if cut_times is None: cut_times = []

  ds1 = pygeons.diff.acc()  
  ds2 = pygeons.diff.disp_laplacian()
  ds = [ds1,ds2]

  out = {}
  out['time'] = data['time']
  out['logitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']

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

  for dir in ['east','north','vertical']:
    u = data[dir]
    sigma = data[dir+'_std']
    out[dir] = pygeons.smooth.smooth(data['time'],pos,u,
                                     sigma=sigma,diff_specs=ds,
                                     time_scale=time_scale,
                                     length_scale=length_scale,
                                     fill=fill)
    out[dir+'_std'] = np.zeros(u.shape)    

  return out
           

def diff(data,time_diff=None,space_diff=None,
         cut_endpoint1_lons=None,
         cut_endpoint1_lats=None,
         cut_endpoint2_lons=None,
         cut_endpoint2_lats=None,
         cut_times=None):

  if cut_endpoint1_lons is None: cut_endpoint1_lons = []
  if cut_endpoint1_lats is None: cut_endpoint1_lats = []
  if cut_endpoint2_lons is None: cut_endpoint2_lons = []
  if cut_endpoint2_lats is None: cut_endpoint2_lats = []
  if cut_times is None: cut_times = []
  
  out = {}
  out['time'] = data['time']
  out['logitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']

  # form DiffSpecs instance
  ds = pygeons.diff.DiffSpecs()
  ds['time']['diffs'] = [time_diff]
  ds['space']['diffs'] = [space_diff]

  bm = _make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T

  cut_endpoints1 = [bm(*i) for i,j in zip(cut_endpoint1_lons,cut_endpoint1_lats)]
  cut_endpoints2 = [bm(*i) for i,j in zip(cut_endpoint2_lons,cut_endpoint2_lats)]
  ds = pygeons.diff.DiffSpecs()
  ds['time']['cuts'] = pygeons.cuts.TimeCuts(cut_times)
  ds['space']['cuts'] = pygeons.cuts.SpaceCuts(cut_endpoints1,cut_endpoints2)

  for dir in ['east','north','vertical']:
    u = data[dir]
    mask = np.isinf(data[dir+'_std'])
    out[dir] = pygeons.diff.diff(data['time'],pos,u,ds,mask=mask)
    out[dir+'_std'] = np.zeros(u.shape)    

  return out
  

def clean(data,resolution='i',**kwargs):
  out = {}
  out['time'] = data['time']
  out['logitude'] = data['longitude']
  out['latitude'] = data['latitude'] 
  out['id'] = data['id']
  
  fig,ax = plt.sub_plots()
  bm = _make_basemap(data['longitude'],data['latitude'],
                     resolution=resolution,ax=ax)
  bm.drawcountries()
  bm.drawstates() 
  bm.drawcoastlines()
  
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  t = data['time']
  u = data['east']
  v = data['north']
  z = data['vertical']
  su = data['east_std']
  sv = data['north_std']
  sz = data['vertical_std']
  
  out = pygeons.clean.clean(
          t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
          converter=bm,map_ax=ax,**kwargs)

  out['east'] = out[0]          
  out['north'] = out[1]          
  out['vertical'] = out[2]          
  out['east_std'] = out[3]   
  out['north_std'] = out[4]          
  out['vertical_std'] = out[5]   

  return out


def zero(data,**kwargs):
  return

def view(data_list,resolution='i',**kwargs):
  t = data_list[0]['time']
  lon = data_list[0]['longitude']
  lat = data_list[0]['latitude']
  u = [d['east'] for d in data_list]
  v = [d['north'] for d in data_list]
  z = [d['vertical'] for d in data_list]
  su = [d['east_std'] for d in data_list]
  sv = [d['north_std'] for d in data_list]
  sz = [d['vertical_std'] for d in data_list]

  fig,ax = plt.sub_plots()
  bm = _make_basemap(lon,lat,
                     resolution=resolution,ax=ax)
  bm.drawcountries()
  bm.drawstates() 
  bm.drawcoastlines()
  x,y = bm(lon,lat)
  pos = np.array([x,y]).T
  
  pygeons.view.view(
    t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
    converter=bm,map_ax=ax,**kwargs)

  return
