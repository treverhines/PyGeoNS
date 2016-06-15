#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
import pygeons.cuts
import modest.mp
import scipy.sparse
import copy

class DiffSpecs(dict):
  ''' 
  specialized dictionary-like class containing the specifications 
  used to make a differentiation matrix

  A DiffSpec instance contains the dictionaries 'time' and 'space' 
  which contain specifications for differentiation along the time and 
  space dimensions.  Each dictionary contains the following items:  
  
    basis : rbf.basis.RBF instance (default=rbf.basis.phs3)
      radial basis function used to generate the weights. this does 
      nothing if 'diff_type' is poly

    stencil_size : int (default=None)
      number of nodes to use for each finite difference stencil
      
    order : int (default=None)
      polynomial order to use when computing weights.  this does 
      nothing if 'diff_type' is poly
      
    cuts : TimeCutCollection or SpaceCutCollection (default=None)
      indicates discontinuities that should not be smoothed across

    diffs : (N,D) array (default=[[0]](time) or [[0,0]](space))
      derivative orders for each dimension for each term in a 
      differential operator

    coeffs : (N,) array (default=[1.0](time) or [1.0,1.0](space))
      coefficient for each term in a differential operator

    diff_type : str (default='poly'(time) or 'rbf'(space))
      indicates whether to compute weights using an RBF or polynomial 
      expansion. must be either 'rbf' or 'poly'
  

  Note
  ----
    elements in a DiffSpecs instance can be modified just like 
    elements in a dictionary.  When using a DiffSpecs instance as an 
    argument to diff_matrix, diff, or network_smooth, it must contain 
    a 'time' and 'space' dictionary and each dictionary must have 
    entries for 'diffs' and 'coeffs'

    value which default to None do not need to be specified. If left 
    as None they will be assigned values during lower level function 
    calls
    
  '''
  def __init__(self,time=None,space=None):  
    ''' 
    creates a instance containing the minimum specs necessary to 
    form a differentiation matrix
    
    Parameters
    ----------
      time : dict, optional
      
      space : dict, optional
      
    '''
    if time is None:
      time = {}
    if space is None:
      space = {}  

    dict.__init__(self)

    self['time'] = {'basis':rbf.basis.phs3,
                    'stencil_size':None, 
                    'order':None,
                    'cuts':None,
                    'diffs':[[0]],
                    'coeffs':[1.0],
                    'diff_type':'poly'} 
    self['space'] = {'basis':rbf.basis.phs3,
                     'stencil_size':None, 
                     'order':None,
                     'cuts':None,
                     'diffs':[[0,0]],
                     'coeffs':[1.0],
                     'diff_type':'rbf'} 

    self['time'].update(time)                     
    self['space'].update(time)                     

  def __str__(self):
    out = 'DiffSpec instance\n'
    out +=           '    time differentiation specifications\n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['time'].iteritems())
    out +=           '    space differentiation specifications\n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['space'].iteritems())
    return out
    
  def __repr__(self):
    return self.__str__()    

# make a few default DiffSpec instances
DISPLACEMENT_LAPLACIAN = DiffSpecs()
DISPLACEMENT_LAPLACIAN['time']['diffs'] = [[0]]
DISPLACEMENT_LAPLACIAN['time']['coeffs'] = [1.0]
DISPLACEMENT_LAPLACIAN['time']['diff_type'] = 'poly'
DISPLACEMENT_LAPLACIAN['space']['diffs'] = [[2,0],[0,2]]
DISPLACEMENT_LAPLACIAN['space']['coeffs'] = [1.0,1.0]
DISPLACEMENT_LAPLACIAN['space']['diff_type'] = 'rbf'

VELOCITY_LAPLACIAN = DiffSpecs()
VELOCITY_LAPLACIAN['time']['diffs'] = [[1]]
VELOCITY_LAPLACIAN['time']['coeffs'] = [1.0]
VELOCITY_LAPLACIAN['time']['diff_type'] = 'poly'
VELOCITY_LAPLACIAN['space']['diffs'] = [[2,0],[0,2]]
VELOCITY_LAPLACIAN['space']['coeffs'] = [1.0,1.0]
VELOCITY_LAPLACIAN['space']['diff_type'] = 'rbf'

ACCELERATION_LAPLACIAN = DiffSpecs()
ACCELERATION_LAPLACIAN['time']['diffs'] = [[2]]
ACCELERATION_LAPLACIAN['time']['coeffs'] = [1.0]
ACCELERATION_LAPLACIAN['time']['diff_type'] = 'poly'
ACCELERATION_LAPLACIAN['space']['diffs'] = [[2,0],[0,2]]
ACCELERATION_LAPLACIAN['space']['coeffs'] = [1.0,1.0]
ACCELERATION_LAPLACIAN['space']['diff_type'] = 'rbf'

DISPLACEMENT_DX = DiffSpecs()
DISPLACEMENT_DX['time']['diffs'] = [[0]]
DISPLACEMENT_DX['time']['coeffs'] = [1.0]
DISPLACEMENT_DX['time']['diff_type'] = 'poly'
DISPLACEMENT_DX['space']['diffs'] = [[1,0]]
DISPLACEMENT_DX['space']['coeffs'] = [1.0]
DISPLACEMENT_DX['space']['diff_type'] = 'rbf'

VELOCITY_DX = DiffSpecs()
VELOCITY_DX['time']['diffs'] = [[1]]
VELOCITY_DX['time']['coeffs'] = [1.0]
VELOCITY_DX['time']['diff_type'] = 'poly'
VELOCITY_DX['space']['diffs'] = [[1,0]]
VELOCITY_DX['space']['coeffs'] = [1.0]
VELOCITY_DX['space']['diff_type'] = 'rbf'

ACCELERATION_DX = DiffSpecs()
ACCELERATION_DX['time']['diffs'] = [[2]]
ACCELERATION_DX['time']['coeffs'] = [1.0]
ACCELERATION_DX['time']['diff_type'] = 'poly'
ACCELERATION_DX['space']['diffs'] = [[1,0]]
ACCELERATION_DX['space']['coeffs'] = [1.0]
ACCELERATION_DX['space']['diff_type'] = 'rbf'

DISPLACEMENT_DY = DiffSpecs()
DISPLACEMENT_DY['time']['diffs'] = [[0]]
DISPLACEMENT_DY['time']['coeffs'] = [1.0]
DISPLACEMENT_DY['time']['diff_type'] = 'poly'
DISPLACEMENT_DY['space']['diffs'] = [[0,1]]
DISPLACEMENT_DY['space']['coeffs'] = [1.0]
DISPLACEMENT_DY['space']['diff_type'] = 'rbf'

VELOCITY_DY = DiffSpecs()
VELOCITY_DY['time']['diffs'] = [[1]]
VELOCITY_DY['time']['coeffs'] = [1.0]
VELOCITY_DY['time']['diff_type'] = 'poly'
VELOCITY_DY['space']['diffs'] = [[0,1]]
VELOCITY_DY['space']['coeffs'] = [1.0]
VELOCITY_DY['space']['diff_type'] = 'rbf'

ACCELERATION_DY = DiffSpecs()
ACCELERATION_DY['time']['diffs'] = [[2]]
ACCELERATION_DY['time']['coeffs'] = [1.0]
ACCELERATION_DY['time']['diff_type'] = 'poly'
ACCELERATION_DY['space']['diffs'] = [[0,1]]
ACCELERATION_DY['space']['coeffs'] = [1.0]
ACCELERATION_DY['space']['diff_type'] = 'rbf'

DISPLACEMENT = DiffSpecs()
DISPLACEMENT['time']['diffs'] = [[0]]
DISPLACEMENT['time']['coeffs'] = [1.0]
DISPLACEMENT['time']['diff_type'] = 'poly'
DISPLACEMENT['space']['diffs'] = [[0,0]]
DISPLACEMENT['space']['coeffs'] = [1.0]
DISPLACEMENT['space']['diff_type'] = 'rbf'

VELOCITY = DiffSpecs()
VELOCITY['time']['diffs'] = [[1]]
VELOCITY['time']['coeffs'] = [1.0]
VELOCITY['time']['diff_type'] = 'poly'
VELOCITY['space']['diffs'] = [[0,0]]
VELOCITY['space']['coeffs'] = [1.0]
VELOCITY['space']['diff_type'] = 'rbf'

ACCELERATION = DiffSpecs()
ACCELERATION['time']['diffs'] = [[2]]
ACCELERATION['time']['coeffs'] = [1.0]
ACCELERATION['time']['diff_type'] = 'poly'
ACCELERATION['space']['diffs'] = [[0,0]]
ACCELERATION['space']['coeffs'] = [1.0]
ACCELERATION['space']['diff_type'] = 'rbf'

def make_displacement_laplacian_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT_LAPLACIAN)

def make_displacement_dx_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT_DX)

def make_displacement_dy_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT_DY)

def make_displacement_dz_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT_DZ)

def make_velocity_dx_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY_DX)

def make_velocity_dy_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY_DY)

def make_velocity_dz_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY_DZ)

def make_velocity_laplacian_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY_LAPLACIAN)

def make_acceleration_laplacian_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(ACCELERATION_LAPLACIAN)

def make_displacement_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT)

def make_velocity_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY)

def make_acceleration_diff_specs():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(ACCELERATION)


def diff_matrix(t,x,ds,procs=None):
  ''' 
  returns a matrix that performs the specified differentiation of 
  displacement 
  
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    ds : DiffSpec instance
    
    procs : int, optional
    
  '''
  Dt = _time_diff_matrix(t,x,procs=procs,**ds['time'])
  Dx = _space_diff_matrix(t,x,procs=procs,**ds['space'])
  D = Dt.dot(Dx)
  D.eliminate_zeros()
  return D

def _time_diff_matrix(t,x,
                      basis=rbf.basis.phs3,
                      stencil_size=None,
                      order=None,
                      cuts=None,
                      procs=None,
                      diffs=None,
                      coeffs=None,
                      diff_type='rbf'):
  # fill in missing arguments
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]

  if stencil_size is None:
    stencil_size = rbf.fd._default_stencil_size(t[:,None],diffs=diffs)

  if cuts is None:
    cuts = pygeons.cuts.TimeCutCollection()

  if diffs is None:
    raise ValueError('diffs was not specified')

  if coeffs is None:
    raise ValueError('coeffs was not specified')
    
  # return an identity matrix if all derivatives in diffs are zero
  if np.all(np.array(diffs) == 0):
    coeff = np.sum(coeffs)
    Lmaster = coeff*scipy.sparse.eye(Nt*Nx).tocsr()
    Lmaster.tocsr()
    return Lmaster
  
  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  if diff_type == 'rbf':
    basis(np.zeros((0,1)),np.zeros((0,1)),diff=(0,))
    for d in diffs:
      basis(np.zeros((0,1)),np.zeros((0,1)),diff=d)

  def args_maker():
    for xi in x:
      vert,smp = cuts.get_vert_smp(xi)
      # the content of args will be pickled and sent over to each subprocess
      args = (t[:,None],
              basis,
              stencil_size,
              order,
              diffs,
              coeffs,
              vert,
              smp,
              diff_type)
      yield args

  def mappable_diff_matrix(args):
    t = args[0]
    basis = args[1]
    stencil_size = args[2]
    order = args[3]
    diffs = args[4]
    coeffs = args[5]
    vert = args[6]
    smp = args[7]
    diff_type = args[8]
    if diff_type == 'rbf': 
      return rbf.fd.diff_matrix(
               t,
               N=stencil_size,
               diffs=diffs,
               coeffs=coeffs,
               basis=basis,
               order=order,
               vert=vert,
               smp=smp)
    elif diff_type == 'poly':               
      return rbf.fd.poly_diff_matrix(
               t,
               N=stencil_size,
               diffs=diffs,
               coeffs=coeffs,
               vert=vert,
               smp=smp)
    else:
      raise ValueError('diff_type must be rbf or poly')               

  Lsubs = modest.mp.parmap(mappable_diff_matrix,args_maker(),workers=procs)

  # combine submatrices into the master matrix
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))

  rows = np.zeros((stencil_size*Nt,Nx))
  cols = np.zeros((stencil_size*Nt,Nx))
  vals = np.zeros((stencil_size*Nt,Nx))
  for i,Li in enumerate(Lsubs):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[:,i] = wrapped_indices[ri,i]
    cols[:,i] = wrapped_indices[ci,i]
    vals[:,i] = vi

  rows = rows.ravel()
  cols = cols.ravel()
  vals = vals.ravel()

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def _space_diff_matrix(t,x,
                       basis=rbf.basis.phs3,
                       stencil_size=None,
                       order=None,
                       cuts=None,
                       procs=None,
                       diffs=None,
                       coeffs=None,
                       diff_type='rbf'):
  # fill in missing arguments
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]

  if stencil_size is None:
    stencil_size = rbf.fd._default_stencil_size(x,diffs=diffs)

  if cuts is None:
    cuts = pygeons.cuts.SpaceCutCollection()

  if diffs is None:
    raise ValueError('diffs was not specified')

  if coeffs is None:
    raise ValueError('coeffs was not specified')
    
  # return an identity matrix if all derivatives in diffs are zero
  if np.all(np.array(diffs) == 0):
    coeff = np.sum(coeffs)
    Lmaster = coeff*scipy.sparse.eye(Nt*Nx).tocsr()
    Lmaster.tocsr()
    return Lmaster

  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  if diff_type == 'rbf':
    basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,0))
    for d in diffs:
      basis(np.zeros((0,2)),np.zeros((0,2)),diff=d)


  # make submatrices for time smoothing for each station
  def args_maker():
    for ti in t:
      vert,smp = cuts.get_vert_smp(ti)
      # the content of args will be pickled and sent over to each subprocess
      args = (x,
              basis,
              stencil_size,
              order,
              diffs,
              coeffs,
              vert,
              smp,
              diff_type)
      yield args

  def mappable_diff_matrix(args):
    x = args[0]
    basis = args[1]
    stencil_size = args[2]
    order = args[3]
    diffs = args[4]
    coeffs = args[5]
    vert = args[6]
    smp = args[7]
    diff_type = args[8]
    if diff_type == 'rbf': 
      return rbf.fd.diff_matrix(
               x,
               N=stencil_size,
               diffs=diffs,
               coeffs=coeffs,
               basis=basis,
               order=order,
               vert=vert,
               smp=smp)
    elif diff_type == 'poly':               
      return rbf.fd.poly_diff_matrix(
               x,
               N=stencil_size,
               diffs=diffs,
               coeffs=coeffs,
               vert=vert,
               smp=smp)
    else:
      raise ValueError('diff_type must be rbf or poly')               

  Lsubs = modest.mp.parmap(mappable_diff_matrix,args_maker(),workers=procs)

  # combine submatrices into the master matrix
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))

  rows = np.zeros((stencil_size*Nx,Nt))
  cols = np.zeros((stencil_size*Nx,Nt))
  vals = np.zeros((stencil_size*Nx,Nt))
  for i,Li in enumerate(Lsubs):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[:,i] = wrapped_indices[i,ri]
    cols[:,i] = wrapped_indices[i,ci]
    vals[:,i] = vi

  rows = rows.ravel()
  cols = cols.ravel()
  vals = vals.ravel()

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def diff(u,t,x,ds,procs=None):
  ''' 
  differentiates u
  
  Parameters
  ----------
    u : (Nt,Nx) array or (K,Nt,Nx) array

    t : (Nt,) array
    
    x : (Nx,2) array
    
    ds: DiffSpec instance
    
  '''
  t = np.asarray(t)
  x = np.asarray(x)
  u = np.asarray(u)

  Nt = t.shape[0]
  Nx = x.shape[0]

  input2d = False
  if len(u.shape) == 2:
    input2d = True
    if u.shape != (Nt,Nx):
      raise ValueError('u must either be a (Nt,Nx) or (K,Nt,Nx) array')

    u = u.reshape((1,Nt,Nx))

  if u.shape[1:] != (Nt,Nx):
    raise ValueError('u must either be a (Nt,Nx) or (K,Nt,Nx) array')

  D = diff_matrix(t,x,ds,procs=procs)
  u_flat = u.reshape((u.shape[0],Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T

  if input2d:
    u_diff = u_diff_flat.reshape((Nt,Nx))
  else:
    u_diff = u_diff_flat.reshape((u.shape[0],Nt,Nx))

  return u_diff

