#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
import pygeons.cuts
import modest.mp
import scipy.sparse
import copy

class DiffSpecs:
  ''' 
  This class is a container for all the arguments that go into 
  creating a differentiation matrix 
  
  items in self.time get passed to _time_diff_matrix and items in 
  self.space get passed to _space_diff_matrix
  '''
  def __init__(self,time=None,space=None):  
    if time is None:
      time = {}
    if space is None:
      space = {}  
          
    self.time = {}  
    self.time['basis'] = rbf.basis.phs3
    self.time['stencil_size'] = None
    self.time['order'] = None
    self.time['cuts'] = None
    self.time['procs'] = None
    self.time['diffs'] = [[0]]
    self.time['coeffs'] = [1.0,1.0]
    self.time['diff_type'] = 'poly'
    self.time.update(time)
    
    self.space = {}
    self.space['basis'] = rbf.basis.phs3
    self.space['stencil_size'] = None
    self.space['order'] = None
    self.space['cuts'] = None
    self.space['procs'] = None
    self.space['diffs'] = [[0,0]]
    self.space['coeffs'] = [1.0,1.0]
    self.space['diff_type'] = 'rbf'
    self.space.update(time)

  def __str__(self):
    out = 'DiffSpec instance\n'
    out +=           '    time differentiation specifications\n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self.time.iteritems())
    out +=           '    space differentiation specifications\n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self.space.iteritems())
    return out
    
  def __repr__(self):
    return self.__str__()    

# make a few default DiffSpec instances
DISPLACEMENT_LAPLACIAN = DiffSpecs()
DISPLACEMENT_LAPLACIAN.time['diffs'] = [[0]]
DISPLACEMENT_LAPLACIAN.time['coeffs'] = [1.0]
DISPLACEMENT_LAPLACIAN.time['diff_type'] = 'poly'
DISPLACEMENT_LAPLACIAN.space['diffs'] = [[2,0],[0,2]]
DISPLACEMENT_LAPLACIAN.space['coeffs'] = [1.0,1.0]
DISPLACEMENT_LAPLACIAN.space['diff_type'] = 'rbf'

VELOCITY_LAPLACIAN = DiffSpecs()
VELOCITY_LAPLACIAN.time['diffs'] = [[1]]
VELOCITY_LAPLACIAN.time['coeffs'] = [1.0]
VELOCITY_LAPLACIAN.time['diff_type'] = 'poly'
VELOCITY_LAPLACIAN.space['diffs'] = [[2,0],[0,2]]
VELOCITY_LAPLACIAN.space['coeffs'] = [1.0,1.0]
VELOCITY_LAPLACIAN.space['diff_type'] = 'rbf'

ACCELERATION_LAPLACIAN = DiffSpecs()
ACCELERATION_LAPLACIAN.time['diffs'] = [[2]]
ACCELERATION_LAPLACIAN.time['coeffs'] = [1.0]
ACCELERATION_LAPLACIAN.time['diff_type'] = 'poly'
ACCELERATION_LAPLACIAN.space['diffs'] = [[2,0],[0,2]]
ACCELERATION_LAPLACIAN.space['coeffs'] = [1.0,1.0]
ACCELERATION_LAPLACIAN.space['diff_type'] = 'rbf'

DISPLACEMENT = DiffSpecs()
DISPLACEMENT.time['diffs'] = [[0]]
DISPLACEMENT.time['coeffs'] = [1.0]
DISPLACEMENT.time['diff_type'] = 'poly'
DISPLACEMENT.space['diffs'] = [[0,0]]
DISPLACEMENT.space['coeffs'] = [1.0]
DISPLACEMENT.space['diff_type'] = 'rbf'

VELOCITY = DiffSpecs()
VELOCITY.time['diffs'] = [[1]]
VELOCITY.time['coeffs'] = [1.0]
VELOCITY.time['diff_type'] = 'poly'
VELOCITY.space['diffs'] = [[0,0]]
VELOCITY.space['coeffs'] = [1.0]
VELOCITY.space['diff_type'] = 'rbf'

ACCELERATION = DiffSpecs()
ACCELERATION.time['diffs'] = [[2]]
ACCELERATION.time['coeffs'] = [1.0]
ACCELERATION.time['diff_type'] = 'poly'
ACCELERATION.space['diffs'] = [[0,0]]
ACCELERATION.space['coeffs'] = [1.0]
ACCELERATION.space['diff_type'] = 'rbf'

def make_displacement_laplacian_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT_LAPLACIAN)

def make_velocity_laplacian_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY_LAPLACIAN)

def make_acceleration_laplacian_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(ACCELERATION_LAPLACIAN)

def make_displacement_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(DISPLACEMENT)

def make_velocity_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(VELOCITY)

def make_acceleration_diffspec():
  ''' 
  returns a copy of the global variable
  '''
  return copy.deepcopy(ACCELERATION)


def _diff_matrix(t,x,ds):
  ''' 
  returns a matrix that performs the specified differentiation of 
  displacement 
  '''
  Dt = _time_diff_matrix(t,x,**ds.time)
  Dx = _space_diff_matrix(t,x,**ds.space)
  D = Dt.dot(Dx)
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


def differentiate(u,t,x,ds):
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

  D = _diff_matrix(t,x,ds)
  u_flat = u.reshape((u.shape[0],Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T

  if input2d:
    u_diff = u_diff_flat.reshape((Nt,Nx))
  else:
    u_diff = u_diff_flat.reshape((u.shape[0],Nt,Nx))

  return u_diff

