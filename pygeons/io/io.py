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


def _change_extension(f,ext):
  return '.'.join(f.split('.')[:-1] + [ext])
  

def pygeons_toh5(input_text_file,file_type='csv',output_file=None):
  ''' 
  converts a text file to an hdf5 file
  '''
  logger.info('Running pygeons toh5 ...')
  data = dict_from_text(input_text_file,parser=file_type)
  if output_file is None:
    output_file = _change_extension(input_text_file,'h5')
    
  hdf5_from_dict(output_file,data)
  

def pygeons_totext(input_file,output_file=None):  
  ''' 
  converts an hdf5 file to a text file
  '''
  logger.info('Running pygeons totext ...')
  data = dict_from_hdf5(input_file)
  if output_file is None:
    output_file = _change_extension(input_file,'csv')
  
  text_from_dict(output_file,data)  


def pygeons_info(input_file):
  ''' 
  prints metadata 
  '''
  logger.info('Running pygeons info ...')
  data_dict = dict_from_hdf5(input_file)
  # put together info string
  units = 'meters**%s days**%s' % (data_dict['space_exponent'],
                                   data_dict['time_exponent'])
  stations = str(len(data_dict['id']))
  times = str(len(data_dict['time']))
  time_range = '%s, %s' % (mjd.mjd_inv(data_dict['time'][0],'%Y-%m-%d'),
                           mjd.mjd_inv(data_dict['time'][-1],'%Y-%m-%d'))
  lon_range = '%s, %s' % (np.min(data_dict['longitude']),
                          np.max(data_dict['longitude']))
  lat_range = '%s, %s' % (np.min(data_dict['latitude']),
                          np.max(data_dict['latitude']))
  station_names = ', '.join(data_dict['id'])
  info_string =''' 
  units : %s
  stations : %s
  times : %s
  time range : %s
  longitude range : %s
  latitude range : %s
  station names : %s
  ''' % (units,stations,times,time_range,lon_range,lat_range,station_names)
  print(info_string)
  
