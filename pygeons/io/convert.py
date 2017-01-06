''' 
module for converting between input/output data formats
'''
import numpy as np
import logging
import h5py
from pygeons.datadict import DataDict
from pygeons.mjd import mjd_inv,mjd
import pygeons.parser 
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
    date_str = mjd_inv(time[i],'%Y-%m-%d')
    out += ('%s, %e, %e, %e, %e, %e, %e\n' % 
            (date_str,data_dict['north'][i],data_dict['east'][i],data_dict['vertical'][i],
             data_dict['north_std'][i],data_dict['east_std'][i],data_dict['vertical_std'][i]))

  return out             
  
def text_from_dict(outfile,data_dict):
  ''' 
  Creates a csv string for every station in *data_dict*. Joins the 
  strings, separated by '***', in *outfiles*
  '''
  Nx = len(data_dict['id'])
  strs = []
  for i in range(Nx):
    # create a subdictionary for each station
    dict_i = {}
    mask = (~np.isfinite(data_dict['north_std'][:,i]) |
            ~np.isfinite(data_dict['east_std'][:,i]) |
            ~np.isfinite(data_dict['vertical_std'][:,i]))
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
  Writes an hdf5 file from the data dictionary
  '''
  fout = h5py.File(outfile,'w') 
  for k in data_dict.keys():
    fout[k] = data_dict[k]
    
  fout.close()
  return


## Load DataDict instances from files
#####################################################################

def dict_from_text(infile,parser):
  ''' 
  Loads a data dictionary from a text file. 
  
  Parameters
  ----------
  infile : str
    input file name
  
  parser : function
    Function from the module *pygeons.parser*. This function should be 
    able to read in a station string and return a dictionary 
    containing "id", "longitude", "latitude", "time", "east", "north", 
    "vertical", "east_std", "north_std", "vertical_std", "time_power", 
    and "space_power". 
    
  '''
  buff = open(infile,'r')
  strs = buff.read().split('***')
  buff.close()

  # dictionaries of data for each station
  dicts = [parser(s) for s in strs]

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
  out['time_power'] = dicts[0]['time_power']
  out['space_power'] = dicts[0]['space_power']
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
    out[key + '_std'] = np.empty((Nt,Nx))
    out[key][:,:] = np.nan
    out[key + '_std'][:,:] = np.inf 
    for i,d in enumerate(dicts):
      idx = [time_dict[t] for t in d['time']]
      out[key][idx,i] = d[key]
      out[key + '_std'][idx,i] = d[key + '_std']

  out = DataDict(out)
  return out


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


