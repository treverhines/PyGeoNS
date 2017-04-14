''' 
Defines the main filtering functions which are called by the 
PyGeoNS executable.
'''
from __future__ import division
import numpy as np
import logging
import os
from pygeons.main.reml import reml
from pygeons.main.strain import strain
from pygeons.mjd import mjd_inv,mjd
from pygeons.basemap import make_basemap
from pygeons.io.convert import dict_from_hdf5,hdf5_from_dict
logger = logging.getLogger(__name__)


def _params_dict(b):
  ''' 
  coerce the list *b* into a dictionary of hyperparameters for each
  direction. The dictionary keys are 'east', 'north', and 'vertical'.
  The dictionary values are each an N array of hyperparameters.
  
    >>> b1 = [1.0,2.0]
    >>> b2 = ['1.0','2.0']
    >>> b3 = ['east','1.0','2.0','north','1.0','2.0','vertical','1.0','2.0']
  
  '''
  b = list(b)
  msg = ('the hyperparameters must be a list of N floats or 3 lists '
         'of N floats where each list is preceded by "east", "north", '
         'or "vertical"')

  if ('east' in b) & ('north' in b) & ('vertical' in b):
    if (len(b) % 3) != 0:
      raise ValueError(msg)

    arr = np.reshape(b,(3,-1))
    dirs = arr[:,0].astype(str) # directions
    vals = arr[:,1:].astype(float) # hyperparameter array
    out = dict(zip(dirs,vals))
    # make sure the keys contain 'east', 'north', and 'vertical'
    if set(out.keys()) != set(['east','north','vertical']):
      raise ValueError(msg)

  else:
    try:
      arr = np.array(b,dtype=float)
    except ValueError:
      raise ValueError(msg)

    out = {'east':arr,
           'north':arr,
           'vertical':arr}

  print(out)
  return out


def _change_extension(f,ext):
  return '.'.join(f.split('.')[:-1] + [ext])


def pygeons_reml(input_file,model,params,fix=(),
                 procs=0,parameters_file=None):
  ''' 
  Restricted maximum likelihood estimation of temporal
  hyperparameters.
  
  Parameters
  ----------
  input_file : str

  model : str
    
  params : (P,) float array
  
  fix : (L,) int array
    Indices of the parameters which will be fixed
    
  procs : int, optional
    Number of subprocesses to spawn.
  
  '''
  data = dict_from_hdf5(input_file)
  if data['time_exponent'] != 0:
    raise ValueError('input dataset must have units of displacement')

  if data['space_exponent'] != 1:
    raise ValueError('input dataset must have units of displacement')

  logger.info('Performing spatial restricted maximum likelihood estimation ...')
  # convert params to a dictionary of hyperparameters for each direction
  params = _params_dict(params)

  # make output file
  if parameters_file is None:
    parameters_file = 'parameters.txt'
    # make sure an existing file is not overwritten
    count = 0
    while os.path.exists(parameters_file):
      count += 1
      parameters_file = 'parameters.%s.txt' % count

  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  xy = np.array([x,y]).T

  for dir in ['east','north','vertical']:
    # optimal hyperparameters for each timeseries, coresponding
    # likelihoods and data counts.
    opt,like,count,param_units = reml(data['time'][:,None], 
                                      xy, 
                                      data[dir], 
                                      data[dir+'_std_dev'],
                                      model,
                                      params[dir],
                                      fix=fix)
                             
    if dir == 'east':                                
      with open(parameters_file,'w') as fout:
        header = '%-15s%-15s' % ('component','count') 
        header += "".join(['%-15s' % ('p%s[%s]' % (j,i)) for j,i in enumerate(param_units)])
        header += '%-15s\n' % 'likelihood'
        fout.write(header)
        fout.flush()
        
    # convert units optimal hyperparameters back to mm and yr   
    with open(parameters_file,'a') as fout:
      entry  = '%-15s%-15s' % (dir,count)
      entry += "".join(['%-15.4e' % j for j in opt])
      entry += '%-15.4e\n' % like
      fout.write(entry)
      fout.flush()


def pygeons_gpr(input_file,prior_model,prior_params,
                start_date=None,stop_date=None,positions=None,
                noise_model='null',noise_params=(),
                return_sample=False,
                outlier_tol=4.0,
                output_file=None):
  ''' 
  Performs temporal Gaussian process regression.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  prior_model : str
    String specifying the prior model
  
  prior_params : float array
    Hyperparameters for the prior  
  
  positions : str, optional
    Position file for the output data set. 

  diff : int, optional
    Derivative order.
  
  noise_model : str
    String specifying the noise model
  
  noise_params : float array
    Hyperparameters for the noise

  return_sample : bool, optional
    Return a sample of the posterior.

  procs : int, optional
    Number of subprocesses to spawn.

  outlier_tol : float, optional
    Tolerance for outlier detection. 

  '''
  data = dict_from_hdf5(input_file)
  if data['time_exponent'] != 0:
    raise ValueError('input dataset must have units of displacement')

  if data['space_exponent'] != 1:
    raise ValueError('input dataset must have units of displacement')
    
  logger.info('Performing spatial Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
  out_dx = dict((k,np.copy(v)) for k,v in data.iteritems())
  out_dy = dict((k,np.copy(v)) for k,v in data.iteritems())

  # convert params to a dictionary of hyperparameters for each direction
  prior_params = _params_dict(prior_params)
  noise_params = _params_dict(noise_params)

  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  xy = np.array([x,y]).T
  # set output positions
  if positions is None:
    output_id = np.array(data['id'],copy=True)
    output_lon = np.array(data['longitude'],copy=True)
    output_lat = np.array(data['latitude'],copy=True)
  else:  
    pos = np.loadtxt(positions,dtype=str)
    # pos = id,longitude,latitude
    output_id = pos[:,0]
    output_lon = pos[:,1].astype(float)
    output_lat = pos[:,2].astype(float)

  output_x,output_y = bm(output_lon,output_lat)
  output_xy = np.array([output_x,output_y]).T 
  
  # set output times
  if start_date is None:
    start_date = mjd_inv(np.min(data['time']),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd_inv(np.max(data['time']),'%Y-%m-%d')
  
  start_time = mjd(start_date,'%Y-%m-%d')  
  stop_time = mjd(stop_date,'%Y-%m-%d')  
  output_time = np.arange(start_time,stop_time+1)
  
  # convert the domain units from m to km
  for dir in ['east','north','vertical']:
    u,su,dx,sdx,dy,sdy  = strain(data['time'],
                                 xy,
                                 data[dir],
                                 data[dir+'_std_dev'],
                                 prior_model,prior_params[dir],
                                 out_t=output_time,
                                 out_x=output_xy,
                                 noise_model=noise_model,
                                 noise_params=noise_params[dir],
                                 tol=outlier_tol,
                                 return_sample=return_sample)
    # convert back to m and days
    out[dir] = u
    out[dir+'_std_dev'] = su
    out_dx[dir] = dx
    out_dx[dir+'_std_dev'] = sdx
    out_dy[dir] = dy
    out_dy[dir+'_std_dev'] = sdy

  # set the new lon lat and id if positions was given
  out['time'] = output_time
  out['longitude'] = output_lon
  out['latitude'] = output_lat
  out['id'] = output_id

  out_dx['time'] = output_time
  out_dx['longitude'] = output_lon
  out_dx['latitude'] = output_lat
  out_dx['id'] = output_id
  out_dx['time_exponent'] = -1
  out_dx['space_exponent'] = 0
  
  out_dy['time'] = output_time
  out_dy['longitude'] = output_lon
  out_dy['latitude'] = output_lat
  out_dy['id'] = output_id
  out_dy['time_exponent'] = -1
  out_dy['space_exponent'] = 0

  # write to output file
  if output_file is None:
    output_file = _change_extension(input_file,'strain.h5')

  output_dx_file = _change_extension(output_file,'dx.h5')
  output_dy_file = _change_extension(output_file,'dy.h5')
  hdf5_from_dict(output_file,out)
  hdf5_from_dict(output_dx_file,out_dx)
  hdf5_from_dict(output_dy_file,out_dy)
  
