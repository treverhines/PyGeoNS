''' 

This module contains functions that check the consistency of data 
dictionaries. A valid dictionary will contain the following entries

  time : (Nt,) array
    Array of observation times. These are integer values of modified 
    Julian dates. Days are used instead of years because there is no 
    ambiguity about the length of a day
        
  id : (Nx,) array
    Array of 4-character IDs for each station

  longitude : (Nx,) array
    Longitude for each station in degrees
        
  latitude : (Nx,) array
    Latitude for each station in degrees

  east,north,vertical : (Nt,Nx) array
    data components. The units should be in terms of meters and days 
    and should be consistent with the values specified for 
    *space_exponent* and *time_exponent*. For example, if 
    *time_exponent* is -1 and *space_exponent* is 1 then the units 
    should be in meters per day.

  east_std,north_std,vertical_std : (Nt,Nx) array
    One standard deviation uncertainty. These should have the same 
    units as the data components

  time_exponent : int
    Indicates the power of the time units for the data. -1 
    indicates that the data is a rate, -2 is an acceleration etc.

  space_exponent : int
    Indicates the power of the spatial units for the data

'''
import copy
import numpy as np
import logging
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
          'vertical','east_std','north_std','vertical_std',
          'time_exponent','space_exponent']
  for k in keys:
    if not self.has_key(k):
      raise DataError('Data dictionary is missing *%s*' % k)
  

def check_shapes(data):
  ''' 
  Checks if the shapes of each entry are consistent.
  '''
  # check for proper sizes
  Nt = self['time'].shape[0]
  Nx = self['id'].shape[0]
  keys = ['longitude','latitude']
  for k in keys:     
    if not self[k].shape == (Nx,):
      raise DataError('*%s* has shape %s but it should have shape %s' 
                      % (k,self[k].shape,(Nx,)))
  
  keys = ['east','north','vertical','east_std','north_std','vertical_std']
  for k in keys:     
    if not self[k].shape == (Nt,Nx):
      raise DataError('*%s* has shape %s but it should have shape %s' 
                      % (k,self[k].shape,(Nt,Nx)))


def check_positive_uncertainties(data):
  ''' 
  Checks if all the uncertainties are positive.
  '''
  keys = ['east_std','north_std','vertical_std']
  for k in keys:
    if np.any(self[k] <= 0.0):
      raise DataError('*%s* contains zeros or negative values' % k)
     

class DataDict(dict):
  ''' 
  Specialized dictionary for GPS data.  An instance of this class 
  should contain the following entries
  
  
  Calling the function *check_self_consistency* will verify that all 
  the entries exist and have the proper shapes 
  '''
  def __init__(self,input):
    ''' 
    Initiates a DataDict instance. 
    
    Parameters
    ----------
      input : dict or DataDict instance
        This should contain all the necessary items for a DataDict 
        instance. Items from *input* are copied so that the new 
        DataDict does not share any data with *input*.

    '''
    dict.__init__(self)
    self.update_with_copy(input)
    return

  def __str__(self):
    out = ('<DataDict : '
           'times = %s, '
           'stations = %s, '
           'units = meters**%s days**%s>'
           % (len(self['time']),len(self['id']),
              self['space_exponent'],self['time_exponent']))
    
    return out

  def __repr__(self):
    return self.__str__()
        
  def update_with_copy(self,other):
    '''makes a deep copy before update'''
    other = copy.deepcopy(other)
    self.update(other)
    
  def check_self_consistency(self):
    ''' 
    raises an error if any of the following conditions are not met

      - no items are None
      - all items have consistent sizes
      - all inf uncertainties are inf for each direction (e.g. if the 
        easting component is unknown then the northing component should 
        also be unknown)
      - all nan data have corresponding uncertainties of inf
      - all uncertainties are positive and nonzero
      - there are no duplicate station IDs

    '''
    logger.debug('checking self consistency ...')
    # check for Nones
    keys = ['time','id','longitude','latitude',
            'east','north','vertical',
            'east_std','north_std','vertical_std',
            'time_exponent','space_exponent']
    for k in keys:
      if not self.has_key(k):
        raise ValueError('*%s* has not been set' % k)

    # check for proper sizes
    Nt = len(self['time'])
    Nx = len(self['id'])
    keys = ['longitude','latitude']
    for k in keys:     
      if not self[k].shape == (Nx,):
        raise ValueError(
          '*%s* has shape %s but should be %s' % (k,self[k].shape,(Nx,)))

    keys = ['east','north','vertical',
            'east_std','north_std','vertical_std']
    for k in keys:     
      if not self[k].shape == (Nt,Nx):
        raise ValueError(
          'dimensions of *%s* have shape %s but should be %s' 
          % (k,self[k].shape,(Nt,Nx)))
    
    # make sure infinite uncertainties are consistent for all three 
    # directions
    east_isinf = np.isinf(self['east_std'])
    north_isinf = np.isinf(self['north_std'])
    vertical_isinf = np.isinf(self['vertical_std'])
    if not np.all(east_isinf==north_isinf):
      raise ValueError(
        'missing data (i.e. data with infinite uncertainty) must be '
        'missing for all three directions')
    if not np.all(east_isinf==vertical_isinf):
      raise ValueError(
        'missing data (i.e. data with infinite uncertainty) must be '
        'missing for all three directions')
      
    # make sure that infs correspond with nans
    keys = ['east','north','vertical']
    for k in keys:
      k_isnan = np.isnan(self[k])
      if not np.all(k_isnan == east_isinf):
        raise ValueError(
          'infs in the uncertainties do not correspond to nans in '
          'the data')
        
    # make sure that all uncertainties are positive
    keys = ['east_std','north_std','vertical_std']
    for k in keys:
      if np.any(self[k] < 0.0):
        raise ValueError(
          'found negative uncertainty for *%s*' % k)
     
    # make sure there are no duplicate station IDs
    if len(np.unique(self['id'])) != len(self['id']):
      # there are duplicate stations, now find which ones
      duplicates = []
      for i in np.unique(self['id']):
        if np.sum(i == self['id']) > 1:
          duplicates += [i]

      duplicates = ' '.join(duplicates)
      raise ValueError(
        'There were multiple occurrences of these station IDs: %s' 
        % duplicates)

    logger.debug('ok')
    return

  def check_compatibility(self,other):
    '''make sure that the DataDicts have the same stations and times'''
    # compare agains the first data set
    if len(self['time']) != len(other['time']):
      raise ValueError('DataDicts have inconsistent number of time epochs')
    if len(self['id']) != len(other['id']):
      raise ValueError('DataDicts have inconsistent number of stations')
    if not np.all(self['time'] == other['time']):
      raise ValueError('DataDicts have inconsistent time epochs')
    if not np.all(self['id'] == other['id']):
      raise ValueError('DataDicts have inconsistent stations')
    if self['time_exponent'] != other['time_exponent']:
      raise ValueError('DataDicts have inconsistent units')
    if self['space_exponent'] != other['space_exponent']:
      raise ValueError('DataDicts have inconsistent units')

    return

