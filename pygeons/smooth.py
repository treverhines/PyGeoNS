#!/usr/bin/env python
import numpy as np
import modest.cv
import rbf.fd
import rbf.poly
import modest
import modest.solvers
import scipy.sparse
import logging
from scipy.spatial import cKDTree
import pygeons.cuts
import modest.mp

logger = logging.getLogger(__name__)


def _identify_duplicate_stations(pos):
  ''' 
  identifies stations which are abnormally close to eachother
  '''
  # if there is zero or one station then dont run this check
  if pos.shape[0] <= 1:
    return

  T = cKDTree(pos)
  dist,idx = T.query(pos,2)
  r = dist[:,1]
  ri = idx[:,1]
  logr = np.log10(r)
  cutoff = np.mean(logr) - 4*np.std(logr)
  duplicates = np.nonzero(logr < cutoff)[0]
  for d in duplicates:
    logger.warning('station %s is close to station %s. '
                   'This may result in numerical instability. One '                 
                   'of the stations should be removed or they should '
                   'be merged together.' % (d,ri[d]))


class _RunningVariance(object):
  ''' 
  estimates uncertainty from bootstrapping without storing all 
  of the bootstrapped solutions

  Usage
  -----
    >> a = _RunningVariance()
    >> a.add(1.8)
    >> a.add(2.3)
    >> a.add(1.5)
    >> a.get_variance()
       0.163333333333334
  '''
  def __init__(self):
    self.sum = None
    self.sumsqs = None
    self.count = 0.0

  def add(self,entry):
    entry = np.asarray(entry,dtype=float)
    self.count += 1.0
    if self.sum is None:
      self.sum = np.copy(entry)
      self.sumsqs = np.copy(entry)**2
    else:
      self.sum += entry
      self.sumsqs += entry**2

  def get_variance(self):
    return (self.sumsqs - self.sum**2/self.count)/(self.count-1.0)


def _bootstrap_uncertainty(G,L,itr=10,**kwargs):
  ''' 
  estimates the uncertainty for the solution to the regularized linear 
  system.  Bootstrapping is necessary because computing the model 
  covariance matrix is too expensive.  It is assumed that G is already 
  weighted by the data uncertainty
  '''
  if itr <= 1:
    logger.info('cannot bootstrap uncertainties with %s iterations. returning zeros' % itr)
    return np.zeros(G.shape[1])

  soln = _RunningVariance()
  for i in range(itr):
    d = np.random.normal(0.0,1.0,G.shape[0])
    solni = modest.sparse_reg_petsc(G,L,d,**kwargs)
    soln.add(solni)
    logger.info('finished bootstrap iteration %s of %s' % (i+1,itr))

  return np.sqrt(soln.get_variance())


def _reg_matrices(t,x,
                  stencil_time_size,
                  stencil_space_size,
                  reg_basis,
                  reg_time_order, 
                  reg_space_order, 
                  time_cuts,
                  space_cuts,
                  procs,
                  baseline):

  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  reg_basis(np.zeros((0,1)),np.zeros((0,1)),diff=(0,))
  reg_basis(np.zeros((0,1)),np.zeros((0,1)),diff=(2,))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,0))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(2,0))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,2))

  # make submatrices for spatial smoothing on each time step
  def space_args_maker():
    for ti in t:
      vert,smp = space_cuts.get_vert_smp(ti) 
      args = (x,stencil_space_size,
              np.array([1.0,1.0]),
              np.array([[2,0],[0,2]]), 
              reg_basis,reg_space_order,
              vert,smp)
      yield args 

  # make submatrices for time smoothing for each station
  def time_args_maker():
    for xi in x:
      vert,smp = time_cuts.get_vert_smp(xi) 
      # note that stencil size and polynomial order are hard coded at 
      # 3 and 2
      args = (t[:,None],stencil_time_size,
              np.array([1.0]),
              np.array([[2]]), 
              reg_basis,reg_time_order,
              vert,smp)
      yield args 

  def mappable_diff_matrix(args):
    return rbf.fd.diff_matrix(args[0],N=args[1],
                              coeffs=args[2],diffs=args[3],
                              basis=args[4],order=args[5],
                              vert=args[6],smp=args[7])

  # form the submatrices in parallel
  Lx = modest.mp.parmap(mappable_diff_matrix,space_args_maker(),Nprocs=procs)
  Lt = modest.mp.parmap(mappable_diff_matrix,time_args_maker(),Nprocs=procs)

  Nx = x.shape[0]
  Nt = t.shape[0]

  ## combine submatrices into the master matrix
  ###################################################################
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
  
  rows = np.zeros((stencil_time_size*Nt,Nx))
  cols = np.zeros((stencil_time_size*Nt,Nx))
  vals = np.zeros((stencil_time_size*Nt,Nx))
  for i,Li in enumerate(Lt):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[:,i] = wrapped_indices[ri,i]
    cols[:,i] = wrapped_indices[ci,i]
    vals[:,i] = vi

  rows = rows.ravel()
  cols = cols.ravel()
  vals = vals.ravel()

  # form sparse time regularization matrix
  Lt_out = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))

  rows = np.zeros((stencil_space_size*Nx,Nt))
  cols = np.zeros((stencil_space_size*Nx,Nt))
  vals = np.zeros((stencil_space_size*Nx,Nt))
  for i,Li in enumerate(Lx):
    Li = Li.tocoo()
    ri,ci,vi = Li.row,Li.col,Li.data
    rows[:,i] = wrapped_indices[i,ri]
    cols[:,i] = wrapped_indices[i,ci]
    vals[:,i] = vi

  rows = rows.ravel()
  cols = cols.ravel()
  vals = vals.ravel()
  
  # form sparse spatial regularization matrix
  Lx_out = scipy.sparse.csr_matrix((vals,(rows,cols)),(Nx*Nt,Nx*Nt))
  

  if baseline:
    # zero all regularization constraints for the first timestep since 
    # all displacements are initially zero
    zero_cols = wrapped_indices[0,:] 

    Lt_out = Lt_out.tocoo()
    Lx_out = Lx_out.tocoo()

    for z in zero_cols:
      Lt_out.data[Lt_out.col==z] = 0.0
      Lx_out.data[Lx_out.col==z] = 0.0

    Lt_out = Lt_out.tocsr()
    Lx_out = Lx_out.tocsr()

  # remove any unnecessary zero entries
  Lt_out.eliminate_zeros()
  Lx_out.eliminate_zeros()

  return Lt_out,Lx_out


def _system_matrix(Nt,Nx,baseline):
  if baseline:
    # estimate baseline value
    diag_data = np.ones(Nx*(Nt - 1))
    diag_row = np.arange(Nx,Nx*Nt)
    diag_col = np.arange(Nx,Nx*Nt)

    bl_data = np.ones(Nx*Nt)
    bl_row = np.arange(Nx*Nt)
    bl_col = np.arange(Nx)[None,:].repeat(Nt,axis=0).flatten()
  
    data = np.concatenate((diag_data,bl_data))
    row = np.concatenate((diag_row,bl_row))
    col = np.concatenate((diag_col,bl_col))
    G = scipy.sparse.csr_matrix((data,(row,col)),(Nt*Nx,Nt*Nx))

  else:
    G = scipy.sparse.eye(Nx*Nt).tocsr()

  return G


def network_smoother(u,t,x,
                     sigma=None,
                     stencil_time_size=None,
                     stencil_time_cuts=None,
                     stencil_space_size=None,
                     stencil_space_cuts=None,
                     reg_basis=rbf.basis.phs3,
                     reg_space_order=None,
                     reg_space_parameter=None,
                     reg_time_order=None,
                     reg_time_parameter=None,
                     solve_ksp='lgmres',
                     solve_pc='icc',
                     solve_max_itr=1000,
                     solve_atol=1e-6, 
                     solve_rtol=1e-8, 
                     solve_view=False, 
                     cv_itr=100,
                     cv_space_bounds=None,
                     cv_time_bounds=None,
                     cv_plot=False,
                     cv_fold=10,
                     cv_chunk='both',
                     bs_itr=10,
                     procs=None,
                     baseline=True):

  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)

  # check for duplicate stations 
  _identify_duplicate_stations(x)

  if cv_space_bounds is None:
    cv_space_bounds = [-4.0,4.0]
  if cv_time_bounds is None:
    cv_time_bounds = [-4.0,4.0]
 
  Nx = x.shape[0]
  Nt = t.shape[0]

  if u.shape != (Nt,Nx):
    raise TypeError('u must have shape (Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))
  
  else:
    sigma = np.array(sigma,copy=True)
    # replace any infinite uncertainties with 1e10. This does not 
    # seem to introduce any numerical instability and seems to be 
    # sufficient in all cases. Infs need to be replaced in order to 
    # keep the LHS matrix positive definite
    sigma[sigma==np.inf] = 1e10
    
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  # make default stencil sizes and poly orders if not specified
  if stencil_space_size is None:
    stencil_space_size = min(x.shape[0],5)

  if stencil_time_size is None:
    stencil_time_size = min(t.shape[0],3)

  if reg_space_order is None:
    # use a polynomial order of 1 unless that is too large for the stencil size
    max_order = rbf.poly.maximum_order(stencil_space_size,2)  
    reg_space_order = min(max_order,1)

  if reg_time_order is None:
    # use a polynomial order of 1 unless that is too large for the stencil size
    max_order = rbf.poly.maximum_order(stencil_time_size,1)  
    reg_time_order = min(max_order,2)

  # make cut collections if not given
  if stencil_time_cuts is None:
    stencil_time_cuts = pygeons.cuts.TimeCutCollection()
  if stencil_space_cuts is None:
    stencil_space_cuts = pygeons.cuts.SpaceCutCollection()


  u_flat = u.flatten()
  sigma_flat = sigma.flatten()

  logger.info('building regularization matrix...')

  Lt,Lx = _reg_matrices(t,x,
                        stencil_time_size,
                        stencil_space_size,
                        reg_basis,
                        reg_time_order, 
                        reg_space_order, 
                        stencil_time_cuts,
                        stencil_space_cuts,
                        procs,baseline)
  logger.info('done')

  logger.info('building system matrix...')
  G = _system_matrix(Nt,Nx,baseline)
  logger.info('done')

  # weigh G and u by the inverse of data uncertainty. this creates 
  # duplicates but G should still be small
  Wdata = 1.0/sigma_flat
  Wrow = range(Nt*Nx)
  Wcol = range(Nt*Nx)
  Wsize = (Nt*Nx,Nt*Nx)
  W = scipy.sparse.csr_matrix((Wdata,(Wrow,Wcol)),Wsize)

  G = W.dot(G)
  u_flat = W.dot(u_flat)

  # clean up any zero entries
  G.eliminate_zeros()

  # make cross validation testing sets if necessary. the testing sets 
  # are split up by station
  if (reg_time_parameter is None) | (reg_space_parameter is None):
    if cv_chunk == 'space':
      cv_fold = min(cv_fold,Nx)
      testing_x_sets = modest.cv.chunkify(range(Nx),cv_fold) 
      data_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
      testing_sets = []
      for tx in testing_x_sets:
        testing_sets += [data_indices[:,tx].flatten()]

    elif cv_chunk == 'time':
      cv_fold = min(cv_fold,Nt)
      testing_t_sets = modest.cv.chunkify(range(Nt),cv_fold) 
      data_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
      testing_sets = []
      for tt in testing_t_sets:
        testing_sets += [data_indices[tt,:].flatten()]

    elif cv_chunk == 'both':
      cv_fold = min(cv_fold,Nt*Nx)
      testing_sets = modest.cv.chunkify(range(Nt*Nx),cv_fold)
 
    else:
      raise ValueError('cv_chunk must be either "space", "time", or "both"') 

  # estimate damping parameters
  if (reg_time_parameter is None) & (reg_space_parameter is None):
    logger.info(
      'damping parameters were not specified and will now be '
      'estimated with cross validation')

    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=testing_sets,
            plot=cv_plot,log_bounds=[cv_time_bounds,cv_space_bounds],
            solver='petsc',ksp=solve_ksp,pc=solve_pc,
            maxiter=solve_max_itr,view=solve_view,atol=solve_atol,
            rtol=solve_rtol,Nprocs=procs)

    reg_time_parameter = out[0][0] 
    reg_space_parameter = out[0][1] 
    
  elif reg_time_parameter is None:
    logger.info(
      'time damping parameter was not specified and will now be '
      'estimated with cross validation')
    if reg_space_parameter == 0.0:
      raise ValueError(
        'space penalty parameter cannot be zero when estimating time '
        'penalty parameter')

    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=testing_sets,plot=cv_plot,
            log_bounds=[cv_time_bounds,
                        [np.log10(reg_space_parameter)-1e-4,
                         np.log10(reg_space_parameter)+1e-4]],
            solver='petsc',ksp=solve_ksp,pc=solve_pc,
            maxiter=solve_max_itr,view=solve_view,atol=solve_atol,
            rtol=solve_rtol,Nprocs=procs)
    reg_time_parameter = out[0][0]

  elif reg_space_parameter is None:
    logger.info(
      'spatial damping parameter was not specified and will now be '
      'estimated with cross validation')
    if reg_time_parameter == 0.0:
      raise ValueError(
        'time penalty parameter cannot be zero when estimating space '
        'penalty parameter')

    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=testing_sets,plot=cv_plot,
            log_bounds=[[np.log10(reg_time_parameter)-1e-4,
                         np.log10(reg_time_parameter)+1e-4],
                        cv_space_bounds],
            solver='petsc',ksp=solve_ksp,pc=solve_pc,
            maxiter=solve_max_itr,view=solve_view,atol=solve_atol,
            rtol=solve_rtol,Nprocs=procs)
    reg_space_parameter = out[0][1]

  # this makes matrix copies
  L = scipy.sparse.vstack((reg_time_parameter*Lt,reg_space_parameter*Lx))

  logger.info('solving for predicted displacements...')
  u_pred = modest.sparse_reg_petsc(G,L,u_flat,
                                   ksp=solve_ksp,pc=solve_pc,
                                   maxiter=solve_max_itr,view=solve_view,
                                   atol=solve_atol,rtol=solve_rtol)
  logger.info('done')

  logger.info('bootstrapping uncertainty...')
  sigma_u_pred = _bootstrap_uncertainty(G,L,itr=bs_itr,
                                        ksp=solve_ksp,pc=solve_pc,
                                        maxiter=solve_max_itr,view=solve_view,
                                        atol=solve_atol,rtol=solve_rtol)
  logger.info('done')

  u_pred = u_pred.reshape((Nt,Nx))
  sigma_u_pred = sigma_u_pred.reshape((Nt,Nx))

  # zero the initial displacements if we are removing baseline
  if baseline:
    u_pred[0,:] = 0.0
    sigma_u_pred[0,:] = 0.0

  return u_pred,sigma_u_pred


