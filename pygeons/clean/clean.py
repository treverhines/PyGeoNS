''' 
Defines cleaning functions which are called by the PyGeoNS executable.
'''
from __future__ import division
import numpy as np
import logging
import matplotlib.pyplot as plt
from pygeons.io.convert import dict_from_hdf5,hdf5_from_dict
from pygeons.mjd import mjd,mjd_inv
from pygeons.basemap import make_basemap
from pygeons.clean.iclean import InteractiveCleaner
from pygeons.units import unit_conversion
from pygeons.plot.plot import (_unit_string,
                               _setup_map_ax,
                               _setup_ts_ax)                               
logger = logging.getLogger(__name__)


def _remove_extension(f):
  '''remove file extension if one exists'''
  if '.' not in f:
    return f
  else:
    return '.'.join(f.split('.')[:-1])


def pygeons_clean(input_file,resolution='i',
                  input_edits_file=None,
                  break_lons=None,break_lats=None,
                  break_conn=None,no_display=False,
                  output_stem=None,**kwargs):
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
  logger.info('Running pygeons clean ...')
  data = dict_from_hdf5(input_file)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  ts_fig,ts_ax = plt.subplots(3,1,sharex=True,num='Time Series View',facecolor='white')
  _setup_ts_ax(ts_ax)
  map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
  bm = make_basemap(data['longitude'],data['latitude'],resolution=resolution)
  _setup_map_ax(bm,map_ax)
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  t = data['time']
  dates = [mjd_inv(ti,'%Y-%m-%d') for ti in t]
  units = _unit_string(data['space_exponent'],data['time_exponent'])
  conv = 1.0/unit_conversion(units,time='day',space='m')
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
        # ignore blank lines
        if line.isspace():
          continue
          
        type,sta,a,b = line.strip().split()
        # set the current station in *ic* to the station for this edit
        xidx, = (data['id'] == sta).nonzero()
        if len(xidx) == 0:
          # continue because the station does not exist in this 
          # dataset
          continue
          
        ic.xidx = xidx[0]
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
    
  # set output file name
  if output_stem is None:
    output_stem = _remove_extension(input_file) + '.clean'

  output_file = output_stem + '.h5'
  output_edits_file = output_stem + '.txt'
  
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
        
  logger.info('Edits saved to %s' % output_edits_file)
  clean_data  = ic.get_data()                 
  out['east'] = clean_data[0]/conv
  out['north'] = clean_data[1]/conv
  out['vertical'] = clean_data[2]/conv
  out['east_std_dev'] = clean_data[3]/conv
  out['north_std_dev'] = clean_data[4]/conv
  out['vertical_std_dev'] = clean_data[5]/conv

  hdf5_from_dict(output_file,out)  
  logger.info('Cleaned data written to %s' % output_file)
  logger.info('Edits written to %s' % output_edits_file)
  return 
