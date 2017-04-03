''' 
This module contains functions that check the consistency of data 
dictionaries. See the README.rst for a description of a valid data 
dictionary.
'''
import numpy as np
import logging
from pygeons.mjd import mjd_inv
logger = logging.getLogger(__name__)

class DataError(Exception):
  ''' 
  An error indicating that the is data dictionary inconsistent.
  '''
  pass


def check_entries(data):
  ''' 
  Checks if all the entries exist in the data dictionary
  '''
  keys = ['time','id','longitude','latitude','east','north',
          'vertical','east_std_dev','north_std_dev','vertical_std_dev',
          'time_exponent','space_exponent']
  for k in keys:
    if not data.has_key(k):
      raise DataError('Data dictionary is missing *%s*' % k)
  

def check_shapes(data):
  ''' 
  Checks if the shapes of each entry are consistent.
  '''
  # check for proper sizes
  Nt = data['time'].shape[0]
  Nx = data['id'].shape[0]
  keys = ['longitude','latitude']
  for k in keys:     
    if not data[k].shape == (Nx,):
      raise DataError('*%s* has shape %s but it should have shape %s' 
                      % (k,data[k].shape,(Nx,)))
  
  keys = ['east','north','vertical',
          'east_std_dev','north_std_dev','vertical_std_dev']
  for k in keys:     
    if not data[k].shape == (Nt,Nx):
      raise DataError('*%s* has shape %s but it should have shape %s' 
                      % (k,data[k].shape,(Nt,Nx)))


def check_positive_uncertainties(data):
  ''' 
  Checks if all the uncertainties are positive.
  '''
  keys = ['east_std_dev','north_std_dev','vertical_std_dev']
  for k in keys:
    if np.any(data[k] < 0.0):
      raise DataError('*%s* contains zeros or negative values' % k)
     

def check_missing_data(data):
  ''' 
  Checks if all nan observations correspond to inf uncertainties and 
  vice versa. If this is not the case then plotting functions may not 
  work properly.
  '''
  dirs = ['east','north','vertical']
  for d in dirs:
    mu = data[d] 
    sigma = data[d + '_std_dev'] 
    is_nan = np.isnan(mu)
    is_inf = np.isinf(sigma)
    if not np.all(is_nan == is_inf):
      raise DataError('nan values in *%s* do not correspond to inf '
                      'values in *%s*' % (d,d+'_std_dev'))
    
    # make sure that there are no nans in the uncertainties or infs in 
    # the means
    is_inf = np.isinf(mu)
    is_nan = np.isnan(sigma)
    if np.any(is_inf):
      raise DataError('inf values found in *%s*' % d)

    if np.any(is_nan):
      raise DataError('nan values found in *%s_std_dev*' % d)
      

def check_unique_stations(data):
  ''' 
  makes sure each station id is unique
  '''
  unique_ids = list(set(data['id']))
  if len(data['id']) != len(unique_ids):
    # there are duplicate stations, now find them
    duplicates = []
    for i in unique_ids:
      if sum(data['id'] == i) > 1:
        duplicates += [i]
        
    duplicates = ', '.join(duplicates)
    raise DataError(
      'Dataset contains the following duplicate station IDs : %s ' 
      % duplicates) 


def check_unique_dates(data):
  ''' 
  makes sure each station id is unique
  '''
  unique_days = list(set(data['time']))
  if len(data['time']) != len(unique_days):
    # there are duplicate dates, now find them
    duplicates = []
    for i in unique_days:
      if sum(data['time'] == i) > 1:
        duplicates += [mjd_inv(i,'%Y-%m-%d')]
        
    duplicates = ', '.join(duplicates)
    raise DataError(
      'Dataset contains the following duplicate dates : %s ' 
      % duplicates) 
  

def check_data(data):
  ''' 
  Runs all data consistency check
  '''
  logger.debug('checking data consistency ... ')
  check_entries(data)
  check_shapes(data)
  check_positive_uncertainties(data)
  check_missing_data(data)
  check_unique_stations(data)
  check_unique_dates(data)
  logger.debug('ok')
  
  
