#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
import pygeons.cuts
import modest.mp
import scipy.sparse

def _time_diff_matrix(t,x,
                      basis=rbf.basis.phs3,
                      stencil_size=None,
                      order=None,
                      cuts=None,
                      procs=None,
                      diff=None,
                      diffs=None,
                      coeffs=None):
  # fill in missing arguments
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]

  if stencil_size is None:
    stencil_size = min(t.shape[0],3)

  if order is None:
    max_order = rbf.poly.maximum_order(stencil_size,1)
    order = min(max_order,2)

  if cuts is None:
    cuts = pygeons.cuts.TimeCutCollection()

  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  basis(np.zeros((0,1)),np.zeros((0,1)),diff=(0,))
  if diffs is not None:
    for d in diffs:
      basis(np.zeros((0,1)),np.zeros((0,1)),diff=d)

  if diff is not None:
    basis(np.zeros((0,1)),np.zeros((0,1)),diff=diff)

  # make submatrices for time smoothing for each station
  def args_maker():
    for xi in x:
      vert,smp = cuts.get_vert_smp(xi)
      args = (t[:,None],stencil_size,
              diff,diffs,coeffs,
              basis,order,
              vert,smp)
      yield args

  def mappable_diff_matrix(args):
    return rbf.fd.diff_matrix(args[0],N=args[1],
                              diff=args[2],diffs=args[3],coeffs=args[4],
                              basis=args[5],order=args[6],
                              vert=args[7],smp=args[8])

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
                       diff=None,
                       diffs=None,
                       coeffs=None):
  # fill in missing arguments
  t = np.asarray(t)
  x = np.asarray(x)
  Nt = t.shape[0]
  Nx = x.shape[0]

  if stencil_size is None:
    stencil_size = min(x.shape[0],5)

  if order is None:
    max_order = rbf.poly.maximum_order(stencil_size,2)
    order = min(max_order,1)

  if cuts is None:
    cuts = pygeons.cuts.SpaceCutCollection()

  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,0))
  if diffs is not None:
    for d in diffs:
      basis(np.zeros((0,2)),np.zeros((0,2)),diff=d)

  if diff is not None:
    basis(np.zeros((0,2)),np.zeros((0,2)),diff=diff)

  # make submatrices for time smoothing for each station
  def args_maker():
    for ti in t:
      vert,smp = cuts.get_vert_smp(ti)
      args = (x,stencil_size,
              diff,diffs,coeffs,
              basis,order,
              vert,smp)
      yield args

  def mappable_diff_matrix(args):
    return rbf.fd.diff_matrix(args[0],N=args[1],
                              diff=args[2],diffs=args[3],coeffs=args[4],
                              basis=args[5],order=args[6],
                              vert=args[7],smp=args[8])

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

def time_diff(u,t,x,**kwargs):
  ''' 
  returns a time derivative of u

  Parameters
  ----------
    u: (Nt,Nx) array or (K,Nt,Nx) array of displacements
    t: times 
    x: station positions
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

  D = _time_diff_matrix(t,x,**kwargs)
  u_flat = u.reshape((u.shape[0],Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T

  if input2d:
    u_diff = u_diff_flat.reshape((Nt,Nx))
  else:
    u_diff = u_diff_flat.reshape((u.shape[0],Nt,Nx))

  return u_diff

def space_diff(u,t,x,**kwargs):
  ''' 
  returns a spatial derivative of u

  Parameters
  ----------
    u: (Nt,Nx) array or (K,Nt,Nx) array of displacements
    t: times 
    x: station positions
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

  D = _space_diff_matrix(t,x,**kwargs)
  u_flat = u.reshape((u.shape[0],Nt*Nx))
  u_diff_flat = D.dot(u_flat.T).T

  if input2d:
    u_diff = u_diff_flat.reshape((Nt,Nx))
  else:
    u_diff = u_diff_flat.reshape((u.shape[0],Nt,Nx))

  return u_diff
