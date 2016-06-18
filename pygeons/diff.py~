#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
import pygeons.cuts
import modest.mp
import modest
import scipy.sparse
import logging
logger = logging.getLogger(__name__)

class DiffSpecs(dict):
  ''' 
  specialized dictionary-like class containing the specifications 
  used to make a differentiation matrix

  A DiffSpecs instance contains the dictionaries 'time' and 'space' 
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

    self['time'] = {'basis':None,
                    'stencil_size':None, 
                    'order':None,
                    'cuts':None,
                    'diffs':None,
                    'coeffs':None,
                    'diff_type':None} 
    self['space'] = {'basis':None,
                     'stencil_size':None, 
                     'order':None,
                     'cuts':None,
                     'diffs':None,
                     'coeffs':None,
                     'diff_type':None} 

    self['time'].update(time)                     
    self['space'].update(time)                     

  def __str__(self):
    out = 'DiffSpecs instance\n'
    out +=           '    time : \n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['time'].iteritems())
    out +=           '    space : \n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['space'].iteritems())
    return out
    
  def __repr__(self):
    return self.__str__()    

  def fill(self,t,x):
    ''' 
    uses the x and t values to fill in any Nones

    Parameters
    ----------
      t : (Nt,) array

      x : (Nx,2) array
    '''    
    Nt = t.shape[0]
    Nx = x.shape[0]
    T = self['time']
    X = self['space']

    if T['basis'] is None:
      T['basis'] = rbf.basis.phs3

    if T['diffs'] is None:
      T['diffs'] = [[0]]

    if T['coeffs'] is None:
      T['coeffs'] = [1.0 for d in T['diffs']]
      
    if T['stencil_size'] is None:
      T['stencil_size'] = rbf.fd._default_stencil_size(Nt,1,diffs=T['diffs'])

    if T['order'] is None:
      T['order'] = rbf.fd._default_poly_order(T['stencil_size'],1)
      
    if T['cuts'] is None:
      T['cuts'] = pygeons.cuts.TimeCuts()  
      
    if T['diff_type'] is None:
      T['diff_type'] = 'poly'  

    if X['basis'] is None:
      X['basis'] = rbf.basis.phs3

    if X['diffs'] is None:
      X['diffs'] = [[0,0]]

    if X['coeffs'] is None:
      X['coeffs'] = [1.0 for d in X['diffs']]
      
    if X['stencil_size'] is None:
      X['stencil_size'] = rbf.fd._default_stencil_size(Nx,2,diffs=X['diffs'])

    if X['order'] is None:
      X['order'] = rbf.fd._default_poly_order(X['stencil_size'],2)
      
    if X['cuts'] is None:
      X['cuts'] = pygeons.cuts.SpaceCuts()  
      
    if X['diff_type'] is None:
      X['diff_type'] = 'rbf'  


# make a convenience functions that generate common diff specs
def disp_laplacian():
  ''' 
  returns displacement laplacian DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[0]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[2,0],[0,2]]
  out['space']['coeffs'] = [1.0,1.0]
  return out

def vel_laplacian():
  ''' 
  returns velocity laplacian DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[1]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[2,0],[0,2]]
  out['space']['coeffs'] = [1.0,1.0]
  return out

def acc_laplacian():
  ''' 
  returns acceleration laplacian DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[2]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[2,0],[0,2]]
  out['space']['coeffs'] = [1.0,1.0]
  return out

def disp_dx():
  ''' 
  returns displacement x derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[0]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[1,0]]
  out['space']['coeffs'] = [1.0]
  return out

def vel_dx():
  ''' 
  returns velocity x derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[1]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[1,0]]
  out['space']['coeffs'] = [1.0]
  return out

def acc_dx():  
  ''' 
  returns acceleration x derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[2]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[1,0]]
  out['space']['coeffs'] = [1.0]
  return out
  
def disp_dy():
  ''' 
  returns displacement y derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[0]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,1]]
  out['space']['coeffs'] = [1.0]
  return out

def vel_dy():
  ''' 
  returns velocity y derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[1]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,1]]
  out['space']['coeffs'] = [1.0]
  return out

def acc_dy():
  ''' 
  returns acceleration y derivative DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[2]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,1]]
  out['space']['coeffs'] = [1.0]
  return out
  
def disp():
  ''' 
  returns displacement DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[0]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,0]]
  out['space']['coeffs'] = [1.0]
  return out

def vel():
  ''' 
  returns velocity DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[1]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,0]]
  out['space']['coeffs'] = [1.0]
  return out
  
def acc():
  ''' 
  returns acceleration DiffSpecs instance
  '''  
  out = DiffSpecs()
  out['time']['diffs'] = [[2]]
  out['time']['coeffs'] = [1.0]
  out['space']['diffs'] = [[0,0]]
  out['space']['coeffs'] = [1.0]
  return out


@modest.funtime
def diff_matrix(t,x,ds,procs=None):
  ''' 
  returns a matrix that performs the specified differentiation of 
  displacement. A differentiation matrix is generated for both time 
  and space and the output of this function is their product. This 
  means that the differential opperator must be expressible as the 
  product of a time and space differential operator. For example:
  
    possible : Dt(Dxx + Dyy)
  
    not possible : Dt*Dxx + Dyy
  
  Parameters
  ----------
    t : (Nt,) array
    
    x : (Nx,2) array
    
    ds : DiffSpecs instance
    
    procs : int, optional
    
  '''
  t = np.asarray(t)
  x = np.asarray(x)
  ds.fill(t,x)
  
  logger.debug('creating differentiation matrix: \n' + str(ds))
  Dt = _time_diff_matrix(t,x,procs=procs,**ds['time'])
  Dx = _space_diff_matrix(t,x,procs=procs,**ds['space'])
  D = Dt.dot(Dx)
  return D

@modest.funtime
def _time_diff_matrix(t,x,
                      basis=None,
                      stencil_size=None,
                      order=None,
                      cuts=None,
                      procs=None,
                      diffs=None,
                      coeffs=None,
                      diff_type=None):
  # fill in missing arguments
  Nt = t.shape[0]
  Nx = x.shape[0]
    
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


@modest.funtime
def _space_diff_matrix(t,x,
                       basis=None,
                       stencil_size=None,
                       order=None,
                       cuts=None,
                       procs=None,
                       diffs=None,
                       coeffs=None,
                       diff_type=None):
  # fill in missing arguments
  Nt = t.shape[0]
  Nx = x.shape[0]
    
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

@modest.funtime
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

