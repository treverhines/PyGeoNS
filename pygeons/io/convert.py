''' 
Module for converting between hdf5 files, text files and data 
dictionaries
'''
import numpy as np
import logging
import h5py
from pygeons.datacheck import check_data
from pygeons.mjd import mjd_inv
from pygeons.io.parser import PARSER_DICT
logger = logging.getLogger(__name__)

## Write files from DataDict instances
#####################################################################
def _write_csv(data):
  ''' 
  Write data for a single station to a csv file
  '''
  time = data['time']
  out  = '4-character id, %s\n' % data['id']
  out += 'begin date, %s\n' % mjd_inv(time[0],'%Y-%m-%d')
  out += 'end date, %s\n' % mjd_inv(time[-1],'%Y-%m-%d')
  out += 'longitude, %s E\n' % data['longitude']
  out += 'latitude, %s N\n' % data['latitude']
  out += ('units, meters**%s days**%s\n' % 
          (data['space_exponent'],data['time_exponent']))
  out += ('date, north, east, vertical, north std. deviation, '
          'east std. deviation, vertical std. deviation\n')
  # convert displacements and uncertainties to strings
  for i in range(len(data['time'])):
    date_str = mjd_inv(time[i],'%Y-%m-%d')
    out += ('%s, %e, %e, %e, %e, %e, %e\n' % 
            (date_str,data['north'][i],data['east'][i],
             data['vertical'][i],data['north_std_dev'][i],
             data['east_std_dev'][i],data['vertical_std_dev'][i]))

  return out             
  
def text_from_dict(outfile,data):
  ''' 
  Writes a text file from a data dictionary. The text file contains a 
  csv string for each station separated by "***".
  
  Parameters
  ----------
  outfile : string
    Name of the output text file

  data : dict
    Data dictionary 

  '''
  logger.info('Converting data dictionary to a text file ...')
  check_data(data)
  Nx = len(data['id'])
  strs = []
  for i in range(Nx):
    # create a subdictionary for each station
    dict_i = {}
    mask = (np.isinf(data['north_std_dev'][:,i]) &
            np.isinf(data['east_std_dev'][:,i]) &
            np.isinf(data['vertical_std_dev'][:,i]))
    # do not write data for this station if the station has no data
    if np.all(mask):
      continue

    dict_i['id'] = data['id'][i]
    dict_i['longitude'] = data['longitude'][i]
    dict_i['latitude'] = data['latitude'][i]
    dict_i['time'] = data['time'][~mask]
    dict_i['east'] = data['east'][~mask,i]
    dict_i['north'] = data['north'][~mask,i]
    dict_i['vertical'] = data['vertical'][~mask,i]
    dict_i['east_std_dev'] = data['east_std_dev'][~mask,i]
    dict_i['north_std_dev'] = data['north_std_dev'][~mask,i]
    dict_i['vertical_std_dev'] = data['vertical_std_dev'][~mask,i]
    dict_i['time_exponent'] = data['time_exponent']
    dict_i['space_exponent'] = data['space_exponent']
    strs += [_write_csv(dict_i)]
    
  out = '***\n'.join(strs)
  fout = open(outfile,'w')
  fout.write(out)
  fout.close()
  return
  

def hdf5_from_dict(outfile,data):
  ''' 
  Writes an hdf5 file from the data dictionary.
  
  Parameters
  ----------
  outfile : str
    Name of the output file
  
  data : dict
    Data dictionary      
  
  '''
  logger.info('Converting data dictionary to an HDF5 file ...')
  check_data(data)
  fout = h5py.File(outfile,'w') 
  for k in data.keys():
    fout[k] = data[k]
    
  fout.close()
  return


## Load DataDict instances from files
#####################################################################
def dict_from_text(infile,parser='csv'):
  ''' 
  Loads a data dictionary from a text file. 
  
  Parameters
  ----------
  infile : str
    Input file name
  
  parser : str
    String indicating which parser to use. Can be either "csv", 
    "pbocsv", "tdecsv", or "pbopos".
    
  Returns
  -------
  out : dict
    Data dictionary
    
  '''
  logger.info('Converting text file to a data dictionary ...')
  buff = open(infile,'r')
  strs = buff.read().split('***')
  buff.close()

  # dictionaries of data for each station
  dicts = [PARSER_DICT[parser](s) for s in strs]

  # find the earliest and latest time. note that these are in MJD
  start_time = np.inf
  stop_time = -np.inf
  for d in dicts:
    if np.min(d['time']) < start_time:
      start_time = np.min(d['time'])
    if np.max(d['time']) > stop_time:
      stop_time = np.max(d['time'])

  # form an array of times ranging from the time of the earliest 
  # observation to the time of the latest observation. count by days
  out = {}
  out['time_exponent'] = dicts[0]['time_exponent']
  out['space_exponent'] = dicts[0]['space_exponent']
  out['time'] = np.arange(int(start_time),int(stop_time)+1,1)
  out['longitude'] = np.array([d['longitude'] for d in dicts])
  out['latitude'] = np.array([d['latitude'] for d in dicts])
  out['id'] = np.array([d['id'] for d in dicts])
  Nt,Nx = len(out['time']),len(out['id'])
  # make a lookup table associating times with indices
  time_dict = dict(zip(out['time'],range(Nt)))
  for key in ['east','north','vertical']:
    # initiate the data arrays with nans or infs. then fill in the 
    # elements where there is
    out[key] = np.empty((Nt,Nx))
    out[key + '_std_dev'] = np.empty((Nt,Nx))
    out[key][:,:] = np.nan
    out[key + '_std_dev'][:,:] = np.inf 
    for i,d in enumerate(dicts):
      idx = [time_dict[t] for t in d['time']]
      out[key][idx,i] = d[key]
      out[key + '_std_dev'][idx,i] = d[key + '_std_dev']

  check_data(out)
  return out


def dict_from_hdf5(infile):
  ''' 
  Loads a data dictionary from an hdf5 file.
  
  Parameters
  ----------
  infile : str
    Name of hdf5 file
    
  Returns
  -------
  out : dict
    Data dictionary

  '''
  logger.info('Converting HDF5 file to a data dictionary ...')
  out = {}
  fin = h5py.File(infile,'r')
  for k in fin.keys():
    out[k] = fin[k][...]

  fin.close()
  check_data(out)
  return out  


