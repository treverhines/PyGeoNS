''' 
Defines the main filtering functions which are called by the 
executables.
'''
from __future__ import division
import numpy as np
import logging
import rbf
import os
from pygeons.filter.gpr import gpr
from pygeons.filter.reml import reml
from pygeons.mjd import mjd_inv,mjd
from pygeons.basemap import make_basemap
from pygeons.breaks import make_time_vert_smp, make_space_vert_smp
logger = logging.getLogger(__name__)


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
                                                

def pygeons_treml(data,model,params,fix=(),order=1,
                  annual=True,semiannual=True,
                  procs=0,output_file=None):
  ''' 
  Restricted maximum likelihood estimation of temporal
  hyperparameters.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  model : str
    Indicates the stochastic model. either 'se', 'fogm', or 'se+fogm'.
    
  params : (P,) float array
    Initial guess for the hyperparameters. length depends on the
    chosen model.
  
  fix : (L,) int array
    Indices of fixed hyperparameters
    
  order : int, optional
    Order of the polynomial basis functions.
  
  no_annual : bool, optional  
    Include annual sinusoids in the noise model.

  no_semiannual : bool, optional  
    Include semiannual sinusoids in the noise model.

  procs : int, optional
    Number of subprocesses to spawn.
  
  output_file : str, optional
    Name of the file that results will be written in.
  
  '''
  logger.info('Performing temporal restricted maximum likelihood estimation ...')
  # make output file
  if output_file is None:
    output_file = 'parameters.txt'
    # make sure an existing file is not overwritten
    count = 0
    while os.path.exists(output_file):
      count += 1
      output_file = 'parameters.%s.txt' % count

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
                                         params,                                
                                         fix=fix,
                                         order=order,
                                         annual=annual,
                                         semiannual=semiannual,
                                         procs=procs)
    if dir == 'east':                                
      with open(output_file,'w') as fout:
        header = '%-15s%-15s%-15s' % ('station','component','count') 
        header += "".join(['%-15s' % i.format(range_units,domain_units) for i in param_units])
        header += '%-15s\n' % 'likelihood'
        fout.write(header)
        fout.flush()
        
    # convert units optimal hyperparameters back to mm and yr   
    with open(output_file,'a') as fout:
      for i,sid in enumerate(data['id']):
        entry  = '%-15s%-15s%-15s' % (sid,dir,counts[i])
        entry += "".join(['%-15.4e' % j for j in opts[i,:]])
        entry += '%-15.4e\n' % likes[i]
        fout.write(entry)
        fout.flush()

    

def pygeons_sreml(data,model,params,fix=(),order=1,
                  procs=0,output_file=None):
  ''' 
  Restricted maximum likelihood estimation of temporal
  hyperparameters.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  model : str
    Covariance function 
    
  params : (P,) float array
    Hyperparameters 
  
  fix : (L,) int array
    Indices of the parameters which will be fixed
    
  order : int, optional
    Order of the polynomial basis functions.
  
  procs : int, optional
    Number of subprocesses to spawn.
  
  '''
  logger.info('Performing spatial restricted maximum likelihood estimation ...')
  # make output file
  if output_file is None:
    output_file = 'parameters.txt'
    # make sure an existing file is not overwritten
    count = 0
    while os.path.exists(output_file):
      count += 1
      output_file = 'parameters.%s.txt' % count

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
                                         params,
                                         fix=fix,
                                         order=order,
                                         procs=procs)
                             
    if dir == 'east':                                
      with open(output_file,'w') as fout:
        header = '%-15s%-15s%-15s' % ('station','component','count') 
        header += "".join(['%-15s' % i.format(range_units,domain_units) for i in param_units])
        header += '%-15s\n' % 'likelihood'
        fout.write(header)
        fout.flush()
        
    # convert units optimal hyperparameters back to mm and yr   
    with open(output_file,'a') as fout:
      for i,sid in enumerate(data['id']):
        entry  = '%-15s%-15s%-15s' % (sid,dir,counts[i])
        entry += "".join(['%-15.4e' % j for j in opts[i,:]])
        entry += '%-15.4e\n' % likes[i]
        fout.write(entry)
        fout.flush()


def pygeons_tgpr(data,prior,order=1,diff=(0,),fogm=(0.5,0.2),
                 annual=True,semiannual=True,
                 outlier_tol=4.0,procs=0,return_sample=False,
                 start_date=None,stop_date=None):
  ''' 
  Performs temporal Gaussian process regression.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  prior : (2,) float array
    hyperparameters for the prior
  
  order : int, optional
    Order of the polynomial basis functions.
  
  diff : int, optional
    Derivative order.
  
  fogm : 2-tuple, optional
    Hyperparameters for the FOGM noise model. 
    
  no_annual : bool, optional  
    Include annual sinusoids in the noise model.

  no_semiannual : bool, optional  
    Include semiannual sinusoids in the noise model.

  outlier_tol : float, optional
    Tolerance for outlier detection.

  procs : int, optional
    Number of subprocesses to spawn.
  
  return_sample : bool, optional
    Returned a sample of the posterior.
    
  start_date : str, optional
    Start date for the output data set.
    
  stop_date : str, optional
    Stop date for the output data set.
    
  '''
  logger.info('Performing temporal Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
  
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
                          prior,
                          x=time[:,None]*domain_conv,
                          order=order,
                          diff=diff,
                          fogm_params=fogm,
                          annual=annual,
                          semiannual=semiannual,
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
  return out
  
def pygeons_sgpr(data,prior,order=1,diff=(0,0),
                 return_sample=False,positions=None,
                 procs=0,outlier_tol=4.0):
  ''' 
  Performs temporal Gaussian process regression.
  
  Parameters
  ----------
  data : dict
    Data dictionary.

  prior : (2,) float array
    Hyperparameters for the prior.
  
  order : int, optional
    Order of the polynomial null space.
  
  diff : int, optional
    Derivative order.
  
  procs : int, optional
    Number of subprocesses to spawn.

  return_sample : bool, optional
    Return a sample of the posterior.

  positions : (str array,float array,float array), optional
    Positions for the output data set. 

  outlier_tol : float, optional
    Tolerance for outlier detection. 

  '''
  logger.info('Performing spatial Gaussian process regression ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())
    
  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  xy = np.array([x,y]).T
  # set output positions
  if positions is None:
    output_id = np.array(data['id'],copy=True)
    output_lon = np.array(data['longitude'],copy=True)
    output_lat = np.array(data['latitude'],copy=True)
  else:  
    output_id,output_lon,output_lat = positions

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
                          prior,
                          x=output_xy*domain_conv,
                          order=order,
                          return_sample=return_sample,
                          diff=diff,
                          tol=outlier_tol,
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
  return out
  

def pygeons_tfilter(data,diff=(0,),fill='none',
                    break_dates=None,**kwargs):
  ''' 
  time smoothing
  '''
  logger.info('Performing temporal RBF-FD filtering ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())  

  vert,smp = make_time_vert_smp(break_dates)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        data['time'][:,None],data[dir].T,
                        sigma=data[dir+'_std_dev'].T,
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,
                        **kwargs)
    out[dir] = post.T
    out[dir+'_std_dev'] = post_sigma.T

  # set the time units
  out['time_exponent'] -= sum(diff)
  return out


def pygeons_sfilter(data,diff=(0,0),fill='none',
                    break_lons=None,break_lats=None,
                    break_conn=None,**kwargs):
  ''' 
  space smoothing
  '''
  logger.info('Performing spatial RBF-FD filtering ...')
  out = dict((k,np.copy(v)) for k,v in data.iteritems())

  bm = make_basemap(data['longitude'],data['latitude'])
  x,y = bm(data['longitude'],data['latitude'])
  pos = np.array([x,y]).T
  vert,smp = make_space_vert_smp(break_lons,break_lats,
                                 break_conn,bm)
  for dir in ['east','north','vertical']:
    post,post_sigma = rbf.filter.filter(
                        pos,data[dir],
                        sigma=data[dir+'_std_dev'],
                        diffs=diff,
                        fill=fill,
                        vert=vert,smp=smp,     
                        **kwargs)
    out[dir] = post
    out[dir+'_std_dev'] = post_sigma

  # set the space units
  out['space_exponent'] -= sum(diff)
  return out

