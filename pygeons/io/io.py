''' 
Defines input/output functions which are called by the PyGeoNS
executables
'''
import numpy as np
from pygeons import mjd
import logging
import warnings
from pygeons.io.convert import (dict_from_hdf5,
                                hdf5_from_dict,
                                text_from_dict,
                                dict_from_text)
logger = logging.getLogger(__name__)                                


def _remove_extension(f):
  '''remove file extension if one exists'''
  if '.' not in f:
    return f
  else:
    return '.'.join(f.split('.')[:-1])  


def _unit_string(space_exponent,time_exponent):
  if space_exponent == 0:
    space_str = '1'
  elif space_exponent == 1:
    space_str = 'm'
  else:
    space_str = 'm^%s' % space_exponent

  if time_exponent == 0:
    time_str = ''
  elif time_exponent == -1:
    time_str = '/day'
  else:
    time_str = '/day^%s' % -time_exponent

  return space_str + time_str
  

def _common_context(data_list):
  ''' 
  Expands the input data dictionaries so that they each have the same 
  context (i.e. stations and observation times).
  '''
  # check for consisten units
  time_exp = data_list[0]['time_exponent']
  if not all(time_exp == d['time_exponent'] for d in data_list):
    raise ValueError('datasets do not have consistent units')

  space_exp = data_list[0]['space_exponent']
  if not all(space_exp == d['space_exponent'] for d in data_list):
    raise ValueError('datasets do not have consistent units')

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
      p[dir] = np.full((Nt,Nx),np.nan)
      p[dir][np.ix_(tidx,sidx)] = d[dir]
      p[dir + '_std_dev'] = np.full((Nt,Nx),np.inf)
      p[dir + '_std_dev'][np.ix_(tidx,sidx)] = d[dir + '_std_dev']

    out_list += [p]

  return out_list
  

def pygeons_merge(input_files,output_stem=None):
  ''' 
  Merge data files
  '''
  logger.info('Running pygeons merge ...')
  data_list = [dict_from_hdf5(i) for i in input_files]
  data_list = _common_context(data_list)
  out = data_list[0]
  for d in data_list[1:]:
    for dir in ['east','north','vertical']:
      # overwrite data in *out* with non-missing data in *d*
      missing_in_d = np.isinf(d[dir+'_std_dev'])
      missing_in_out = np.isinf(out[dir+'_std_dev'])
      if np.any(~missing_in_d & ~missing_in_out):
        warnings.warn(
          'Data for some stations and times exist in multiple '
          'datasets. Precedence is determined by the order the data '
          'files were specified in.')
        
      out[dir][~missing_in_d] = d[dir][~missing_in_d]
      out[dir+'_std_dev'][~missing_in_d] = d[dir+'_std_dev'][~missing_in_d]

  # set output file name
  if output_stem is None:
    output_stem = 'merged'

  output_file = output_stem + '.h5'
  hdf5_from_dict(output_file,out)
  logger.info('Merged data written to %s' % output_file)


def pygeons_crop(input_file,start_date=None,stop_date=None,
                 min_lat=-np.inf,max_lat=np.inf,
                 min_lon=-np.inf,max_lon=np.inf,
                 stations=None,output_stem=None):
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
  logger.info('Running pygeons crop ...')
  data = dict_from_hdf5(input_file)
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  if start_date is None:
    start_date = mjd.mjd_inv(data['time'].min(),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd.mjd_inv(data['time'].max(),'%Y-%m-%d')

  if stations is None:
    stations = []

  # remove times that are not within the bounds of *start_date* and 
  # *stop_date*
  start_time = int(mjd.mjd(start_date,'%Y-%m-%d'))
  stop_time = int(mjd.mjd(stop_date,'%Y-%m-%d'))
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

  # set output file name
  if output_stem is None:
    output_stem = _remove_extension(input_file) + '.crop'

  output_file = output_stem + '.h5'
  hdf5_from_dict(output_file,out)
  logger.info('Cropped data written to %s' % output_file)
  return


def pygeons_toh5(input_text_file,file_type='csv',output_stem=None):
  ''' 
  converts a text file to an hdf5 file
  '''
  logger.info('Running pygeons toh5 ...')
  data = dict_from_text(input_text_file,parser=file_type)
  if output_stem is None:
    output_stem = _remove_extension(input_text_file)

  output_file = output_stem + '.h5'  
  hdf5_from_dict(output_file,data)
  logger.info('Data written to %s' % output_file)
  return
  

def pygeons_totext(input_file,output_stem=None):  
  ''' 
  converts an hdf5 file to a text file
  '''
  logger.info('Running pygeons totext ...')
  data = dict_from_hdf5(input_file)
  if output_stem is None:
    output_stem = _remove_extension(input_file)
  
  output_file = output_stem + '.csv'  
  text_from_dict(output_file,data)  
  logger.info('Data written to %s' % output_file)
  return


def pygeons_info(input_file):
  ''' 
  prints metadata 
  '''
  logger.info('Running pygeons info ...')
  data_dict = dict_from_hdf5(input_file)
  # put together info string
  units = _unit_string(data_dict['space_exponent'],
                       data_dict['time_exponent'])
  stations = str(len(data_dict['id']))
  times = str(len(data_dict['time']))
  observations = (np.sum(~np.isinf(data_dict['east_std_dev'])) +
                  np.sum(~np.isinf(data_dict['north_std_dev'])) +
                  np.sum(~np.isinf(data_dict['vertical_std_dev'])))

  time_range = '%s, %s' % (mjd.mjd_inv(data_dict['time'][0],'%Y-%m-%d'),
                           mjd.mjd_inv(data_dict['time'][-1],'%Y-%m-%d'))
  lon_range = '%s, %s' % (np.min(data_dict['longitude']),
                          np.max(data_dict['longitude']))
  lat_range = '%s, %s' % (np.min(data_dict['latitude']),
                          np.max(data_dict['latitude']))
  # split names into groups of no more than 8
  station_name_list = list(data_dict['id'])
  station_name_groups = []
  while len(station_name_list) > 0:
    station_name_groups += [', '.join(station_name_list[:7])]
    station_name_list = station_name_list[7:]

  msg  = '\n'
  msg += '------------------ PYGEONS DATA INFORMATION ------------------\n\n'
  msg += 'file : %s\n' % input_file
  msg += 'units : %s\n' % units
  msg += 'stations : %s\n' % stations
  msg += 'times : %s\n' % times
  msg += 'observations : %s\n' % observations
  msg += 'time range : %s\n' % time_range
  msg += 'longitude range : %s\n' % lon_range
  msg += 'latitude range : %s\n' % lat_range
  msg += 'station names : %s\n' % station_name_groups[0]
  for g in station_name_groups[1:]:
    msg += '                %s\n' % g

  msg += '\n'  
  msg += '--------------------------------------------------------------'
  print(msg)
  return
  
