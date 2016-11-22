''' 
module for converting between input/output data formats
'''
import numpy as np
import logging
import h5py
from pygeons.mean import MeanInterpolant
from pygeons.datadict import DataDict
import pygeons.parser 
from pygeons.dateconv import decday_inv
logger = logging.getLogger(__name__)

## Write files from DataDict instances
#####################################################################

def _write_csv(data_dict):
  ''' 
  Writes to a PyGeoNS csv file
  '''
  time = data_dict['time']
  out  = '4-character id, %s\n' % data_dict['id']
  out += 'begin date, %s\n' % decday_inv(time[0],'%Y-%m-%d')
  out += 'end date, %s\n' % decday_inv(time[-1],'%Y-%m-%d')
  out += 'longitude, %s E\n' % data_dict['longitude']
  out += 'latitude, %s N\n' % data_dict['latitude']
  out += 'units, meters**%s * days**%s\n' % (data_dict['space_power'],data_dict['time_power'])
  out += 'date, north, east, vertical, north std. deviation, east std. deviation, vertical std. deviation\n'
  # convert displacements and uncertainties to strings
  for i in range(len(data_dict['time'])):
    date_str = decday_inv(time[i],'%Y-%m-%d')
    out += ('%s, %e, %e, %e, %e, %e, %e\n' % 
            (date_str,data_dict['north'][i],data_dict['east'][i],data_dict['vertical'][i],
             data_dict['north_std'][i],data_dict['east_std'][i],data_dict['vertical_std'][i]))

  return out             
  
def csv_from_dict(outfile,data_dict):
  Nx = len(data_dict['id'])
  strs = []
  for i in range(Nx):
    # create a subdictionary for each station
    dict_i = {}
    mask = np.isinf(data_dict['north_std'][:,i])
    # do not write data for this station if the station has no data
    if np.all(mask):
      continue

    dict_i['id'] = data_dict['id'][i]
    dict_i['longitude'] = data_dict['longitude'][i]
    dict_i['latitude'] = data_dict['latitude'][i]
    dict_i['time'] = data_dict['time'][~mask]
    dict_i['east'] = data_dict['east'][~mask,i]
    dict_i['north'] = data_dict['north'][~mask,i]
    dict_i['vertical'] = data_dict['vertical'][~mask,i]
    dict_i['east_std'] = data_dict['east_std'][~mask,i]
    dict_i['north_std'] = data_dict['north_std'][~mask,i]
    dict_i['vertical_std'] = data_dict['vertical_std'][~mask,i]
    dict_i['time_power'] = data_dict['time_power']
    dict_i['space_power'] = data_dict['space_power']
    strs += [_write_csv(dict_i)]
    
  out = '***\n'.join(strs)
  fout = open(outfile,'w')
  fout.write(out)
  fout.close()
  return
  

def hdf5_from_dict(outfile,data_dict):
  ''' 
  writes an hdf5 file from the data dictionary
  '''
  fout = h5py.File(outfile,'w') 
  for k in data_dict.keys():
    fout[k] = data_dict[k]
    
  fout.close()
  return


## Load DataDict instances from files
#####################################################################

def _dict_from_text(infile,file_type):
  ''' 
  loads a data dictionary from a text file. 
  '''
  buff = open(infile,'r')
  strs = buff.read().split('***')
  buff.close()

  # dictionaries of data for each station
  if file_type == 'pbocsv':
    dicts = [pygeons.parser.parse_pbocsv(s) for s in strs]
  if file_type == 'tdecsv':
    dicts = [pygeons.parser.parse_tdecsv(s) for s in strs]
  elif file_type == 'pbopos':
    dicts = [pygeons.parser.parse_pbopos(s) for s in strs]
  elif file_type == 'csv':
    dicts = [pygeons.parser.parse_csv(s) for s in strs]

  # find the start and end time
  start_time = np.inf
  stop_time = -np.inf
  for d in dicts:
    if np.min(d['time']) < start_time:
      start_time = np.min(d['time'])
    if np.max(d['time']) > stop_time:
      stop_time = np.max(d['time'])

  # interpolation times, make sure that the stop_time is included
  time = np.arange(start_time,stop_time+1,1)
  longitude = np.array([d['longitude'] for d in dicts])
  latitude = np.array([d['latitude'] for d in dicts])
  id = np.array([d['id'] for d in dicts])
  Nt,Nx = len(time),len(id)
  east = np.zeros((Nt,Nx))
  north = np.zeros((Nt,Nx))
  vertical = np.zeros((Nt,Nx))
  east_std = np.zeros((Nt,Nx))
  north_std = np.zeros((Nt,Nx))
  vertical_std = np.zeros((Nt,Nx))
  # interpolate data onto the interpolation times for each station
  time_power = dicts[0]['time_power']
  space_power = dicts[0]['space_power']
  for i,d in enumerate(dicts):
    logger.debug('interpolating data for station %s onto grid times' % d['id'])
    data_i = np.concatenate((d['east'][None,:],
                             d['north'][None,:],
                             d['vertical'][None,:]),axis=0)
    sigma_i = np.concatenate((d['east_std'][None,:],
                              d['north_std'][None,:],
                              d['vertical_std'][None,:]),axis=0)
    itp = MeanInterpolant(d['time'][:,None],data_i,sigma_i)
    data_itp,sigma_itp = itp(time[:,None])
    east[:,i] = data_itp[0,:]
    north[:,i] = data_itp[1,:]
    vertical[:,i] = data_itp[2,:]
    east_std[:,i] = sigma_itp[0,:]
    north_std[:,i] = sigma_itp[1,:]
    vertical_std[:,i] = sigma_itp[2,:]

  out = {}
  out['time'] = time
  out['longitude'] = longitude
  out['latitude'] = latitude
  out['id'] = id
  out['east'] = east
  out['north'] = north
  out['vertical'] = vertical
  out['east_std'] = east_std
  out['north_std'] = north_std
  out['vertical_std'] = vertical_std
  out['time_power'] = time_power
  out['space_power'] = space_power
  out = DataDict(out)
  return out


def dict_from_csv(infile):
  return _dict_from_text(infile,'csv')


def dict_from_pbocsv(infile):
  return _dict_from_text(infile,'pbocsv')


def dict_from_pbopos(infile):
  return _dict_from_text(infile,'pbopos')


def dict_from_tdecsv(infile):
  return _dict_from_text(infile,'tdecsv')


def dict_from_hdf5(infile):
  ''' 
  loads a data dictionary from an hdf5 file
  '''
  out = {}
  fin = h5py.File(infile,'r')
  for k in fin.keys():
    out[k] = fin[k][...]

  out = DataDict(out)
  fin.close()
  return out  


