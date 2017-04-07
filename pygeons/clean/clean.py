''' 
Defines functions which are called by the PyGeoNS executable. These 
functions are for data cleaning.
'''
from __future__ import division
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from pygeons.io.convert import dict_from_hdf5,hdf5_from_dict
from pygeons.mjd import mjd,mjd_inv
from pygeons.basemap import make_basemap
from pygeons.clean.iclean import InteractiveCleaner
from pygeons.breaks import make_space_vert_smp
from pygeons.plot.plot import (_unit_string,
                               _unit_conversion,
                               _setup_map_ax,
                               _setup_ts_ax)                               
logger = logging.getLogger(__name__)


def _change_extension(f,ext):
  return '.'.join(f.split('.')[:-1] + [ext])
  

def pygeons_crop(input_file,start_date=None,stop_date=None,
                 min_lat=-np.inf,max_lat=np.inf,
                 min_lon=-np.inf,max_lon=np.inf,
                 stations=None,output_file=None):
  ''' 
  Sets the time span of the data set to be between *start_date* and 
  *stop_date*. Sets the stations to be within the latitude and 
  longitude bounds.
  
  Parameters
  ----------
  data : dict
    data dictionary
      
  start_date : str, optional
    start date of output data set in YYYY-MM-DD. Uses the start date 
    of *data* if not provided. Defaults to the earliest date.

  stop_date : str, optional
    Stop date of output data set in YYYY-MM-DD. Uses the stop date 
    of *data* if not provided. Defaults to the latest date.
      
  min_lon, max_lon, min_lat, max_lat : float, optional
    Spatial bounds on the output data set
  
  stations : str list, optional
    List of stations to be removed from the dataset. This is in 
    addition to the station removed by the lon/lat bounds.
    
  Returns
  -------
  out_dict : dict
    output data dictionary

  '''
  data = dict_from_hdf5(input_file)
  logger.info('Cropping data set ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  if start_date is None:
    start_date = mjd_inv(data['time'].min(),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd_inv(data['time'].max(),'%Y-%m-%d')
  
  if stations is None:
    stations = []  

  # remove times that are not within the bounds of *start_date* and 
  # *stop_date*
  start_time = int(mjd(start_date,'%Y-%m-%d'))
  stop_time = int(mjd(stop_date,'%Y-%m-%d'))
  idx = ((data['time'] >= start_time) &
         (data['time'] <= stop_time))
  out['time'] = out['time'][idx]         
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][idx,:]
    out[dir + '_std_dev'] = out[dir + '_std_dev'][idx,:]
    
  # find stations that are within the bounds
  in_bounds = ((data['longitude'] > min_lon) &
               (data['longitude'] < max_lon) &
               (data['latitude'] > min_lat) &
               (data['latitude'] < max_lat))
  # find stations that are in the list of stations to be removed                
  in_list = np.array([i in stations for i in data['id']])
  # keep stations that are in bounds and not in the list
  idx, = (in_bounds & ~in_list).nonzero()
         
  out['id'] = out['id'][idx]
  out['longitude'] = out['longitude'][idx]
  out['latitude'] = out['latitude'][idx]
  for dir in ['east','north','vertical']:
    out[dir] = out[dir][:,idx]
    out[dir + '_std_dev'] = out[dir + '_std_dev'][:,idx]
    
  if output_file is None:
    output_file = _change_extension(input_file,'crop.h5')
  
  hdf5_from_dict(output_file,out)  
  return 


def pygeons_clean(input_file,resolution='i',
                  input_edits_file=None,
                  output_edits_file=None,
                  break_lons=None,break_lats=None,
                  break_conn=None,no_display=False,
                  output_file=None,**kwargs):
  ''' 
  runs the PyGeoNS Interactive Cleaner
  
  Parameters
  ----------
    data : dict
      data dictionary

    resolution : str
      basemap resolution    
    
    input_edits_file : str
      Name of the file containing edits which will automatically be 
      applied before opening up the interactive viewer.
    
    output_edits_file : str
      Name of the file where all edits will be recorded.   
      
    **kwargs : 
      gets passed to pygeons.clean.clean
         
  Returns
  -------
    out : dict
      output data dictionary 
    
  '''
  data = dict_from_hdf5(input_file)
  logger.info('Cleaning data set ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax)
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(data['longitude'],data['latitude'],resolution=resolution)
  _setup_map_ax(bm,map_ax)
  # draw breaks if there are any
  vert,smp = make_space_vert_smp(break_lons,break_lats,break_conn,bm)
  for s in smp:
    map_ax.plot(vert[s,0],vert[s,1],'k--',lw=2,zorder=2)

  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  t = data['time']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  conv = _unit_conversion(data['space_exponent'],data['time_exponent'])
  units = _unit_string(data['space_exponent'],data['time_exponent'])
  u = conv*data['east']
  v = conv*data['north']
  z = conv*data['vertical']
  su = conv*data['east_std_dev']
  sv = conv*data['north_std_dev']
  sz = conv*data['vertical_std_dev']
  ic = InteractiveCleaner(
         t,pos,u=u,v=v,z=z,su=su,sv=sv,sz=sz,
         map_ax=map_ax,ts_ax=ts_ax,
         time_labels=dates,
         units=units,
         station_labels=data['id'],
         **kwargs)

  # make edits to the data set prior to displaying it
  if input_edits_file is not None:
    with open(input_edits_file,'r') as fin:
      for line in fin: 
        type,sta,a,b = line.strip().split()
        # set the current station in *ic* to the station for this edit
        xidx, = (data['id'] == sta).nonzero()
        if len(xidx) == 0:
          # continue because the station does not exist in this 
          # dataset
          continue
          
        ic.config['xidx'] = xidx[0]
        if type == 'outliers':
          start_time = mjd(a,'%Y-%m-%d')
          stop_time = mjd(b,'%Y-%m-%d')
          ic.remove_outliers(start_time,stop_time)
        elif type == 'jump':
          jump_time = mjd(a,'%Y-%m-%d')
          delta = int(b)
          ic.remove_jump(jump_time,delta)
        else:
          raise ValueError('edit type must be either "outliers" or "jump"')

  if not no_display:
    ic.update()
    ic.connect()
    
  # save edits to output file
  if output_edits_file is None:
    output_edits_file = 'edits.txt'
    # make sure an existing file is not overwritten
    count = 0
    while os.path.exists(output_edits_file):
      count += 1
      output_edits_file = 'edits.%s.txt' % count
  
  with open(output_edits_file,'w') as fout:
    for i in ic.log:
      type,xidx,a,b = i
      if type == 'outliers':
        station = data['id'][xidx]
        start_date = mjd_inv(a,'%Y-%m-%d')
        stop_date = mjd_inv(b,'%Y-%m-%d')
        fout.write('outliers %s %s %s\n' % (station,start_date,stop_date))
      elif type == 'jump':
        station = data['id'][xidx]
        jump_date = mjd_inv(a,'%Y-%m-%d')
        fout.write('jump     %s %s %s\n' % (station,jump_date,b))
      else:
        raise ValueError('edit type must be either "outliers" or "jump"')
        
  logger.info('edits saved to %s' % output_edits_file)
  clean_data  = ic.get_data()                 
  out['east'] = clean_data[0]/conv
  out['north'] = clean_data[1]/conv
  out['vertical'] = clean_data[2]/conv
  out['east_std_dev'] = clean_data[3]/conv
  out['north_std_dev'] = clean_data[4]/conv
  out['vertical_std_dev'] = clean_data[5]/conv

  if output_file is None:
    output_file = _change_extension(input_file,'clean.h5')
  
  hdf5_from_dict(output_file,out)  
  return 
