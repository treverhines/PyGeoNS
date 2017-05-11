''' 
Tools for creating Gaussian processes.
'''
import numpy as np
import scipy.sparse as sp
from rbf.gauss import (GaussianProcess,
                       _get_arg_count,
                       _zero_mean,
                       _zero_covariance,
                       _empty_basis)
from pygeons.units import unit_conversion as conv
from pygeons.main.cbasis import add_diffs_to_caches
import logging
logger = logging.getLogger(__name__)


def station_sigma_and_p(gp,time,mask):
  ''' 
  Build the sparse covariance matrix and basis vectors describing
  noise that is uncorrelated between stations. The covariance and
  basis functions will only be evauluated at unmasked data.
  '''
  logger.debug('Building station covariance matrix and basis '
               'vectors ...')
  diff = np.array([0])             
  sigma_i = gp._covariance(time,time,diff,diff)
  p_i = gp._basis(time,diff)

  Nt,Np = p_i.shape
  _,Nx = mask.shape

  # break sigma_i into data,rows,cols
  if sp.issparse(sigma_i):
    sigma_i = sigma_i.tocoo()
    data_i = sigma_i.data
    rows_i = sigma_i.row
    cols_i = sigma_i.col
  
  else:
    # the matrix is dense
    data_i = sigma_i.ravel()
    rows_i,cols_i = np.mgrid[:Nt,:Nt]
    rows_i = rows_i.ravel()
    cols_i = cols_i.ravel()
  
  # build the data for the sparse covariance matrix for all data
  # (including the masked ones) then crop out the masked ones
  # afterwards. This is not efficient if there are many masked
  # observations but I cannot think of a better way to do this.
  Nnz, = data_i.shape  # number of non-zeros in sigma_i
  p = np.zeros((Nt,Nx,Np,Nx),dtype=float)
  data = np.zeros((Nx,Nnz),dtype=float)  
  rows = np.zeros((Nx,Nnz),dtype=np.int32)  
  cols = np.zeros((Nx,Nnz),dtype=np.int32)  
  for i in range(Nx):
    data[i,:] = data_i
    rows[i,:] = i + rows_i*Nx
    cols[i,:] = i + cols_i*Nx
    p[:,i,:,i] = p_i

  data = data.ravel()
  rows = rows.ravel()
  cols = cols.ravel()
  p = p.reshape((Nt*Nx,Np*Nx))

  if Nx*Nnz > (0.5*Nt**2*Nx**2):
    # if the matrix has more than 50% non-zeros then make the output
    # matrix dense
    sigma = np.zeros((Nt*Nx,Nt*Nx))
    sigma[rows,cols] = data
    logger.debug('Station covariance matrix is dense')
  
  else:
    # otherwise make it csc sparse    
    sigma = sp.csc_matrix((data,(rows,cols)),(Nt*Nx,Nt*Nx),dtype=float)
    density = (100.0*sigma.nnz)/np.prod(sigma.shape)
    logger.debug('Station covariance matrix is sparse with %.3f%% '
                 'non-zeros' % density)

  # toss out rows and columns for masked data
  maskf = mask.ravel()
  sigma = sigma[:,~maskf][~maskf,:]
  p = p[~maskf,:]

  if p.size != 0:
    # remove singluar values from p
    u,s,_ = np.linalg.svd(p,full_matrices=False)
    keep = s > 1e-12*s.max()
    p = u[:,keep]
    logger.debug('Removed %s singular values from the station basis '
                 'vectors' % np.sum(~keep))

  logger.debug('Done')
  return sigma,p


def chunkify_covariance(cov_in,chunk_size):
  ''' 
  Wraps covariance functions so that the covariance matrix is built in
  chunks rather than all at once. This is more memory efficient if the
  covariance function generates multiple intermediary arrays. 
  '''
  def cov_out(x1,x2,diff1,diff2):
    N1,N2 = x1.shape[0],x2.shape[0]
    # Collect the data in data,rows,cols format. Then covert to the
    # proper type at the end
    data = np.zeros((0,),dtype=float)  
    rows = np.zeros((0,),dtype=np.int32)  
    cols = np.zeros((0,),dtype=np.int32)  
    # count is the total number of rows added to the output covariance
    # matrix thus far
    count = 0 
    while count != N1:
      if N1 > chunk_size:
        # only log the progress if the covariance matrix takes
        # multiple chunks to build
        logger.debug(
          'Building covariance matrix (chunk size = %s) : %5.1f%% '
          'complete' % (chunk_size,(100.0*count)/N1))

      start,stop = count,min(count+chunk_size,N1)
      cov_chunk = cov_in(x1[start:stop],x2,diff1,diff2) 
      if sp.issparse(cov_chunk):
        # if sparse convert to coo and get the data
        cov_chunk = cov_chunk.tocoo()
        data = np.hstack((data,cov_chunk.data))
        rows = np.hstack((rows,start + cov_chunk.row))
        cols = np.hstack((cols,cov_chunk.col))

      else:
        # if dense unravel cov_chunk
        r,c = np.mgrid[start:stop,:N2]
        data = np.hstack((data,cov_chunk.ravel()))
        rows = np.hstack((rows,r.ravel()))
        cols = np.hstack((cols,c.ravel()))
        
      count = min(count+chunk_size,N1)
      
    # Decide whether to make the output array sparse or dense based on
    # the number of non-zeros. I could have alternatively had the
    # output mimic the input covariance function.
    Nnz, = data.shape
    if Nnz > (0.5*N1*N2):
      # if the matrix has more than 50% non-zeros then make the output
      # matrix dense
      out = np.zeros((N1,N2))
      out[rows,cols] = data
  
    else:
      # otherwise make it csc sparse    
      out = sp.csc_matrix((data,(rows,cols)),(N1,N2),dtype=float)
    
    if N1 > chunk_size:
      logger.debug(
        'Building covariance matrix (chunk size = %s) : 100.0%% '
        'complete' % chunk_size)
      
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
    cov1  = gp1._covariance(x1[:,[0]],x2[:,[0]],
                            diff1[[0]],diff2[[0]])
    cov2  = gp2._covariance(x1[:,[1,2]],x2[:,[1,2]],
                            diff1[[1,2]],diff2[[1,2]])
    # There are two conditions to consider: (1) cov1 and cov2 are
    # dense, and (2) at least one of the matrices is sparse. If (1)
    # then use *np.multiply* for element-wise multiplication. If (2)
    # then use the *multiply* method of the sparse matrix.
    if (not sp.issparse(cov1)) & (not sp.issparse(cov2)):
      # both are dense. The output will be a dense array
      out = np.multiply(cov1,cov2)

    else:
      # at least one is sparse. THe output will be a sparse array
      if sp.issparse(cov1):
        out = cov1.multiply(cov2).tocsc()
      else:
        # cov2 is sparse
        out = cov2.multiply(cov1).tocsc()
             
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
  
  gp._covariance = chunkify_covariance(gp._covariance,1000)
  return gp
  

