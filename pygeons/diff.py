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
    a 'time' and 'space' dictionary 
    
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
    self['space'].update(space)

  def __str__(self):
    out = 'DiffSpecs instance\n'
    out +=           '    time : \n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['time'].iteritems())
    out +=           '    space : \n'    
    out += ''.join('        %s : %s\n' % (k,v) for (k,v) in self['space'].iteritems())
    return out
    
  def __repr__(self):
    return self.__str__()    

  def fill(self):
    ''' 
    replaces Nones with default values
    '''
    T = self['time']
    X = self['space']

    if T['basis'] is None:
      T['basis'] = rbf.basis.phs3

    if T['diffs'] is None:
      T['diffs'] = [[0]]

    if T['coeffs'] is None:
      T['coeffs'] = [1.0 for d in T['diffs']]
      
    if T['stencil_size'] is None:
      T['stencil_size'] = rbf.fd._default_stencil_size(1,diffs=T['diffs'])

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
      X['stencil_size'] = rbf.fd._default_stencil_size(2,diffs=X['diffs'])

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


def diff_matrix(t,x,ds,procs=None,mask=None):
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
    
    mask : (Nt,Nx) bool array, optional
      If provided then the times and positions where the mask is True 
      will be ignored 
      
  '''
  t = np.asarray(t)
  x = np.asarray(x)
  ds.fill()
  
  if mask is None:
    mask = np.zeros((t.shape[0],x.shape[0]),dtype=bool)
  else:
    mask = np.asarray(mask,dtype=bool)
      
  logger.debug('creating differentiation matrix : \n' + str(ds))
  Dt = _time_diff_matrix(t,x,procs=procs,mask=mask,**ds['time'])
  Dx = _space_diff_matrix(t,x,procs=procs,mask=mask,**ds['space'])
  D = Dt.dot(Dx)
  logger.debug('done')
  
  return D


def _time_diff_matrix(t,x,
                      basis=None,
                      stencil_size=None,
                      order=None,
                      cuts=None,
                      procs=None,
                      diffs=None,
                      coeffs=None,
                      diff_type=None,
                      mask=None):
  # fill in missing arguments
  Nt = t.shape[0]
  Nx = x.shape[0]
    
  # A time differentiation matrix is created for each station in this 
  # loop. If two stations have the same mask and the same time cuts 
  # then reused the matrix. 
  cache = {}
  Lsubs = []
  for i,xi in enumerate(x):
    # create a tuple identifying the time cut and mask for this station 
    cut_indices_tpl = tuple(cuts.get_indices(xi))
    mask_tpl = tuple(mask[:,i])
    key = cut_indices_tpl + mask_tpl
    if key in cache:
      Lsubs += [cache[key]]
      continue

    # find the indices of unmasked times for this station
    sub_idx, = np.nonzero(~mask[:,i])

    vert,smp = cuts.get_vert_and_smp(xi)
    if diff_type == 'rbf': 
      Li = rbf.fd.diff_matrix(
             t[sub_idx,None],N=stencil_size,
             diffs=diffs,coeffs=coeffs,
             basis=basis,order=order,
             vert=vert,smp=smp)

    elif diff_type == 'poly':               
      Li = rbf.fd.poly_diff_matrix(
             t[sub_idx,None],N=stencil_size,
             diffs=diffs,coeffs=coeffs,
             vert=vert,smp=smp)

    else:
      raise ValueError('diff_type must be "rbf" or "poly"')
      
    # convert to coo to get row and col indices for each entry
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    # changes the coordinates to correspond with the t vector rather 
    # than t[sub_idx]
    Li = scipy.sparse.csr_matrix((vi,(sub_idx[ri],sub_idx[ci])),shape=(Nt,Nt))

    cache[key] = Li             
    Lsubs += [Li]
    
  # combine submatrices into the master matrix
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
  
  rows = np.zeros((0,),dtype=int)
  cols = np.zeros((0,),dtype=int)
  vals = np.zeros((0,),dtype=float)
  for i,Li in enumerate(Lsubs):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows = np.concatenate((rows,wrapped_indices[ri,i]))
    cols = np.concatenate((cols,wrapped_indices[ci,i]))
    vals = np.concatenate((vals,vi))

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def _space_diff_matrix(t,x,
                       basis=None,
                       stencil_size=None,
                       order=None,
                       cuts=None,
                       procs=None,
                       diffs=None,
                       coeffs=None,
                       diff_type=None,
                       mask=None):
  # fill in missing arguments
  Nt = t.shape[0]
  Nx = x.shape[0]
    
  # diff matrices for a collection of space cuts are stored in 
  # this dictionary and then recalled if another matrix is to be 
  # generated with the the same space cuts
  cache = {}
  Lsubs = []
  for i,ti in enumerate(t):
    cut_indices_tpl = tuple(cuts.get_indices(ti))
    mask_tpl = tuple(mask[i,:])
    key = cut_indices_tpl + mask_tpl
    if key in cache:
      Lsubs += [cache[key]]
      continue
      
    # find the indices of unmasked stations for this time
    sub_idx, = np.nonzero(~mask[i,:])

    vert,smp = cuts.get_vert_and_smp(ti)
    if diff_type == 'rbf': 
      Li = rbf.fd.diff_matrix(
             x[sub_idx],N=stencil_size,
             diffs=diffs,coeffs=coeffs,
             basis=basis,order=order,
             vert=vert,smp=smp)
      
    elif diff_type == 'poly':               
      Li = rbf.fd.poly_diff_matrix(
             x[sub_idx],N=stencil_size,
             diffs=diffs,coeffs=coeffs,
             vert=vert,smp=smp)
             
    else:
      raise ValueError('diff_type must be rbf or poly')               

    # convert to coo to get row and col indices for each entry
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    # changes the coordinates to correspond with the x vector rather 
    # than x[sub_idx]
    Li = scipy.sparse.csr_matrix((vi,(sub_idx[ri],sub_idx[ci])),shape=(Nx,Nx))

    cache[key] = Li
    Lsubs += [Li]
             
  # combine submatrices into the master matrix
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))

  rows = np.zeros((0,),dtype=int)
  cols = np.zeros((0,),dtype=int)
  vals = np.zeros((0,),dtype=float)
  for i,Li in enumerate(Lsubs):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows = np.concatenate((rows,wrapped_indices[i,ri]))
    cols = np.concatenate((cols,wrapped_indices[i,ci]))
    vals = np.concatenate((vals,vi))

  # form sparse time regularization matrix
  Lmaster = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  return Lmaster


def diff(u,t,x,ds,procs=None,mask=None):
  ''' 
  differentiates u
  
  Parameters
  ----------
    u : (Nt,Nx) array 

    t : (Nt,) array
    
    x : (Nx,2) array
    
    ds : DiffSpec instance
    
    mask : (Nt,Nx) array
      Identifies which elements of u to ignore. This is incase there 
      are outliers or missing data. The returned diffentiated array 
      will have zeros where the mask is True
      
  Returns
  -------
    u_diff : (Nt,Nx) array
    
  '''
  t,x,u = np.asarray(t),np.asarray(x),np.asarray(u)
  Nt,Nx = t.shape[0],x.shape[0]

  D = diff_matrix(t,x,ds,procs=procs,mask=mask)
  u_flat = u.reshape(Nt*Nx)
  u_diff_flat = D.dot(u_flat)
  u_diff = u_diff_flat.reshape((Nt,Nx))

  return u_diff

