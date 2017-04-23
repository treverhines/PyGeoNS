''' 
Defines input/output functions which are called by the PyGeoNS
executables
'''
import numpy as np
from pygeons import mjd
import logging
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
  
