''' 
Tools for creating Gaussian processes.
'''
import numpy as np
from rbf.gauss import GaussianProcess
from rbf.gauss import _get_arg_count,_zero_mean,_zero_covariance,_empty_basis
from pygeons.units import unit_conversion as conv
import logging
from pygeons.main.cbasis import add_diffs_to_caches
logger = logging.getLogger(__name__)


def chunkify_covariance(cov_in,chunk_size):
  ''' 
  Wraps covariance functions so that the covariance matrix is built in
  chunks rather than all at once. This is more memory efficient if the
  covariance function generates multiple intermediary arrays. 
  '''
  def cov_out(x1,x2,diff1,diff2):
    count = 0 
    n1,n2 = x1.shape[0],x2.shape[0]
    out = np.zeros((n1,n2),dtype=float)
    while count < n1:
      # only log the progress if the covariance matrix takes multiple
      # chunks to build
      if n1 > chunk_size:
        logger.debug('Building covariance matrix : %3d%% complete' % ((100.0*count)/n1))
        
      start,stop = count,count+chunk_size
      cov_chunk = cov_in(x1[start:stop],x2,diff1,diff2) 
      out[start:stop] = cov_chunk 
      count += chunk_size
      
    if n1 > chunk_size:
      logger.debug('Building covariance matrix : 100% complete')
      
    return out  
  
  return cov_out      

 
def set_units(units):
  ''' 
  Wrapper for Gaussian process constructors which sets the
  hyperparameter units. When a wrapped constructor is called, the
  hyperparameters are converted to be in terms of *m* and *day*. The
  constructor is also given the key word argument *convert*, which can
  be set to False if no conversion is desired.
  '''
  def decorator(fin):
    def fout(*args,**kwargs):
      convert = kwargs.pop('convert',True)
      if convert:
        args = [a*conv(u,time='day',space='m') for a,u in zip(args,units)]
        
      return fin(*args)

    fout.units = units
    fout.nargs = _get_arg_count(fin)
    fout.__doc__ = fin.__doc__
    fout.__name__ = fin.__name__
    if fout.nargs != len(fout.units):
      raise ValueError(
        'the number of arguments must be equal to the number of unit '
        'specifications') 
    
    return fout  

  return decorator  


def kernel_product(gp1,gp2):
  ''' 
  Returns a GaussianProcess with zero mean and covariance that is the
  product of the two inputs. The first GP must be 1D and the second
  must be 2D.
  '''
  def mean(x,diff):
    return np.zeros(x.shape[0])

  def covariance(x1,x2,diff1,diff2):
    out  = gp1._covariance(x1[:,[0]],x2[:,[0]],
                           diff1[[0]],diff2[[0]])
    out *= gp2._covariance(x1[:,[1,2]],x2[:,[1,2]],
                           diff1[[1,2]],diff2[[1,2]])
    return out
  
  return GaussianProcess(mean,covariance,dim=3)  

  
def null():
  '''   
  returns a GaussianProcess with zero mean and covariance and not
  basis functions
  '''
  return GaussianProcess(_zero_mean,_zero_covariance,_empty_basis)


def composite_units(components,constructors):
  ''' 
  returns the units for the composite Gaussian process hyperparameters
  '''
  components = list(components)
  try:
    cs = [constructors[m] for m in components]
  except KeyError as err:
    raise ValueError(
      '"%s" is not a valid Gaussian process. Use Gaussian processes '
      'from the following list:\n%s' % 
      (err.args[0],', '.join(['"%s"' % i for i in constructors.keys()])))

  units = []
  for ci in cs:
    units += ci.units
  
  return units  


def composite(components,args,constructors):
  ''' 
  Returns a composite Gaussian process. The components are specified
  with *components*, and the arguments for each component are specied
  with *args*. The components must be keys in *constructors*. For
  example,
  
  >>> gpcomp(['fogm','se'],[1.0,2.0,3.0,4.0],gpstation.CONSTRUCTORS) 
  
  creates a GaussianProcess composed of a FOGM and an SE model. The
  first two arguments are passed to *gpfogm* and the second two
  arguments are passed to *se*.
  '''
  # use cythonized functions when evaluating RBFs  
  add_diffs_to_caches()

  components = list(components)
  args = list(args)
  try:
    cs = [constructors[m] for m in components]
  except KeyError as err:
    raise ValueError(
      '"%s" is not a valid Gaussian process. Use Gaussian processes '
      'from the following list:\n%s' % 
      (err.args[0],', '.join(['"%s"' % i for i in constructors.keys()])))
        
  nargs  = sum(ci.nargs for ci in cs)
  if len(args) != nargs:
    raise ValueError(
      '%s parameters were specified for the model "%s", but it '
      'requires %s parameters.\n' %(len(args),' '.join(components),nargs))
  
  gp = null()
  for ci in cs:
    gp += ci(*(args.pop(0) for i in range(ci.nargs)))
  
  gp._covariance = chunkify_covariance(gp._covariance,500)
  return gp
  

