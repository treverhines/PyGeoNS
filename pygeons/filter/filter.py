''' 
Defines the main filtering functions which are called by the 
PyGeoNS executable.
'''
from __future__ import division
import numpy as np
import logging
import os
from pygeons.filter.gpr import gpr
from pygeons.filter.reml import reml
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


def _unit_string(space_exponent,time_exponent):
  ''' 
  returns a string indicating the units. This is different from the
  function in pygeons.plot.plot.py, because it does not convert strain
  to microstrain
  '''
  if space_exponent == 0:
    # if the space exponent is 0 then use units of microstrain
    space_str = '1'
  elif space_exponent == 1:
    space_str = 'mm'
  else:
    space_str = 'mm^%s' % space_exponent

  if time_exponent == 0:
    time_str = ''
  elif time_exponent == -1:
    time_str = '/yr'
  else:
    time_str = '/yr^%s' % -time_exponent

  return space_str + time_str
                                                

def pygeons_treml(input_file,model,params,fix=(),
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

  procs : int, optional
  
  output_file : str, optional
  
  '''
  data = dict_from_hdf5(input_file)
  logger.info('Performing temporal restricted maximum likelihood estimation ...')
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

  # convert the domain units from days to years
  domain_conv = 1.0/365.25 
  domain_units = 'yr'
  # convert the range units from m**a day**b to mm**a yr**b
  range_conv = (1000.0**data['space_exponent'] *
                (1.0/365.25)**data['time_exponent'])
  range_units = _unit_string(data['space_exponent'],data['time_exponent'])                 
  for dir in ['east','north','vertical']:
    # optimal hyperparameters for each timeseries, coresponding
    # likelihoods and data counts.
    opts,likes,counts,param_units = reml(data['time'][:,None]*domain_conv,  
                                         data[dir].T*range_conv,     
                                         data[dir+'_std_dev'].T*range_conv,
                                         model,
                                         params[dir],                                
                                         fix=fix,
                                         procs=procs)
    if dir == 'east':                                
      with open(parameters_file,'w') as fout:
        header = '%-15s%-15s%-15s%-15s%-15s' % ('station','longitude','latitude','component','count') 
        header += "".join(['%-15s' % ('p%s[%s]' % (j,i.format(range_units,domain_units))) for j,i in enumerate(param_units)])
        header += '%-15s\n' % 'likelihood'
        fout.write(header)
        fout.flush()
        
    # convert units optimal hyperparameters back to mm and yr   
    with open(parameters_file,'a') as fout:
      for i,sid in enumerate(data['id']):
        entry  = '%-15s%-15s%-15s%-15s%-15s' % (sid,'%.4f' % data['longitude'][i],'%.4f' % data['latitude'][i],dir,counts[i])
        entry += "".join(['%-15.4e' % j for j in opts[i,:]])
        entry += '%-15.4e\n' % likes[i]
        fout.write(entry)
        fout.flush()

    
def pygeons_sreml(input_file,model,params,fix=(),
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

  # convert the domain units from m to km
  domain_conv = 1.0/1000.0
  domain_units = 'km'
  # convert the range units from m**a day**b to mm**a yr**b
  range_conv = (1000.0**data['space_exponent'] *
                (1.0/365.25)**data['time_exponent'])
  range_units = _unit_string(data['space_exponent'],data['time_exponent'])                 
  for dir in ['east','north','vertical']:
    # optimal hyperparameters for each timeseries, coresponding
    # likelihoods and data counts.
    opts,likes,counts,param_units = reml(xy*domain_conv, 
                                         data[dir]*range_conv, 
                                         data[dir+'_std_dev']*range_conv,
                                         model,
                                         params[dir],
                                         fix=fix,
                                         procs=procs)
                             
    if dir == 'east':                                
      with open(parameters_file,'w') as fout:
        header = '%-15s%-15s%-15s%-15s' % ('date','mjd','component','count') 
        header += "".join(['%-15s' % ('p%s[%s]' % (j,i.format(range_units,domain_units))) for j,i in enumerate(param_units)])
        header += '%-15s\n' % 'likelihood'
        fout.write(header)
        fout.flush()
        
    # convert units optimal hyperparameters back to mm and yr   
    with open(parameters_file,'a') as fout:
      for i,day in enumerate(data['time']):
        date = mjd_inv(day,'%Y-%m-%d')
        entry  = '%-15s%-15s%-15s%-15s' % (date,day,dir,counts[i])
        entry += "".join(['%-15.4e' % j for j in opts[i,:]])
        entry += '%-15.4e\n' % likes[i]
        fout.write(entry)
        fout.flush()


def pygeons_tgpr(input_file,prior_model,prior_params,
                 start_date=None,stop_date=None,diff=(0,),
                 noise_model='null',noise_params=(),
                 outlier_tol=4.0,procs=0,return_sample=False,
                 output_file=None):
  ''' 
  Performs temporal Gaussian process regression.
  
  Parameters
  ----------
  input_file : str

  prior_model : str
    
  prior_params : float array
  
  start_date : str, optional
    
  stop_date : str, optional

  diff : int, optional
  
  noise_model : str, optional
  
  noise_params : float array
    
  outlier_tol : float, optional

  procs : int, optional
  
  return_sample : bool, optional
    
  '''
  data = dict_from_hdf5(input_file)
  logger.info('Performing temporal Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  # convert params to a dictionary of hyperparameters for each direction
  prior_params = _params_dict(prior_params)
  noise_params = _params_dict(noise_params)
  
  # set output times
  if start_date is None:
    start_date = mjd_inv(np.min(data['time']),'%Y-%m-%d')

  if stop_date is None:
    stop_date = mjd_inv(np.max(data['time']),'%Y-%m-%d')
  
  start_time = mjd(start_date,'%Y-%m-%d')  
  stop_time = mjd(stop_date,'%Y-%m-%d')  
  time = np.arange(start_time,stop_time+1)
  # convert the domain units from days to years
  domain_conv = 1.0/365.25 
  # convert the range units from m**a day**b to mm**a yr**b
  range_conv = (1000.0**data['space_exponent'] *
                (1.0/365.25)**data['time_exponent'])
  for dir in ['east','north','vertical']:
    post,post_sigma = gpr(data['time'][:,None]*domain_conv,
                          data[dir].T*range_conv,
                          data[dir+'_std_dev'].T*range_conv,
                          prior_model,prior_params[dir],
                          x=time[:,None]*domain_conv,
                          diff=diff,
                          noise_model=noise_model,
                          noise_params=noise_params[dir],
                          procs=procs,
                          return_sample=return_sample,
                          tol=outlier_tol)
    # convert back to m and days
    post /= (range_conv/domain_conv**sum(diff))                   
    post_sigma /= (range_conv/domain_conv**sum(diff)) 
    out[dir] = post.T
    out[dir+'_std_dev'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  out['time'] = time
  
  # write to output file
  if output_file is None:
    output_file = _change_extension(input_file,'tgpr.h5')
    
  hdf5_from_dict(output_file,out)
  return 
  

def pygeons_sgpr(input_file,prior_model,prior_params,
                 positions=None,diff=(0,0),
                 noise_model='null',noise_params=(),
                 return_sample=False,
                 procs=0,outlier_tol=4.0,
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
  logger.info('Performing spatial Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

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
  # convert the domain units from m to km
  domain_conv = 1.0/1000.0
  # convert the range units from m**a day**b to mm**a yr**b
  range_conv = (1000.0**data['space_exponent'] *
                (1.0/365.25)**data['time_exponent'])
  for dir in ['east','north','vertical']:
    post,post_sigma = gpr(xy*domain_conv,
                          data[dir]*range_conv,
                          data[dir+'_std_dev']*range_conv,
                          prior_model,prior_params[dir],
                          x=output_xy*domain_conv,
                          diff=diff,
                          noise_model=noise_model,
                          noise_params=noise_params[dir],
                          tol=outlier_tol,
                          return_sample=return_sample,
                          procs=procs)
    # convert back to m and days
    post /= (range_conv/domain_conv**sum(diff))                   
    post_sigma /= (range_conv/domain_conv**sum(diff)) 
    out[dir] = post
    out[dir+'_std_dev'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  # set the new lon lat and id if positions was given
  out['longitude'] = output_lon
  out['latitude'] = output_lat
  out['id'] = output_id

  # write to output file
  if output_file is None:
    output_file = _change_extension(input_file,'sgpr.h5')

  hdf5_from_dict(output_file,out)
  return out
  

#def pygeons_tfilter(data,diff=(0,),fill='none',
#                    break_dates=None,**kwargs):
#  ''' 
#  time smoothing
#  '''
#  logger.info('Performing temporal RBF-FD filtering ...')
#  out = dict((k,np.copy(v)) for k,v in data.iteritems())  
#
#  vert,smp = make_time_vert_smp(break_dates)
#  for dir in ['east','north','vertical']:
#    post,post_sigma = rbf.filter.filter(
#                        data['time'][:,None],data[dir].T,
#                        sigma=data[dir+'_std_dev'].T,
#                        diffs=diff,
#                        fill=fill,
#                        vert=vert,smp=smp,
#                        **kwargs)
#    out[dir] = post.T
#    out[dir+'_std_dev'] = post_sigma.T
#
#  # set the time units
#  out['time_exponent'] -= sum(diff)
#  return out
#
#
#def pygeons_sfilter(data,diff=(0,0),fill='none',
#                    break_lons=None,break_lats=None,
#                    break_conn=None,**kwargs):
#  ''' 
#  space smoothing
#  '''
#  logger.info('Performing spatial RBF-FD filtering ...')
#  out = dict((k,np.copy(v)) for k,v in data.iteritems())
#
#  bm = make_basemap(data['longitude'],data['latitude'])
#  x,y = bm(data['longitude'],data['latitude'])
#  pos = np.array([x,y]).T
#  vert,smp = make_space_vert_smp(break_lons,break_lats,
#                                 break_conn,bm)
#  for dir in ['east','north','vertical']:
#    post,post_sigma = rbf.filter.filter(
#                        pos,data[dir],
#                        sigma=data[dir+'_std_dev'],
#                        diffs=diff,
#                        fill=fill,
#                        vert=vert,smp=smp,     
#                        **kwargs)
#    out[dir] = post
#    out[dir+'_std_dev'] = post_sigma
#
#  # set the space units
#  out['space_exponent'] -= sum(diff)
#  return out
#
