#!/usr/bin/env python
import numpy as np
import modest.cv
import rbf.fd
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
                  stencil_size=5,
                  reg_basis=rbf.basis.phs3,
                  reg_poly_order=1, 
                  time_cuts=None,
                  space_cuts=None,
                  procs=None):
  # compile the necessary derivatives for our rbf. This is done so 
  # that each subprocesses does not need to
  reg_basis(np.zeros((0,1)),np.zeros((0,1)),diff=(0,))
  reg_basis(np.zeros((0,1)),np.zeros((0,1)),diff=(2,))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,0))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(2,0))
  reg_basis(np.zeros((0,2)),np.zeros((0,2)),diff=(0,2))

  if time_cuts is None:
    time_cuts = pygeons.cuts.TimeCutCollection()
  if space_cuts is None:
    space_cuts = pygeons.cuts.SpaceCutCollection()

  # make submatrices for spatial smoothing on each time step
  def space_args_maker():
    for ti in t:
      vert,smp = space_cuts.get_vert_smp(ti) 
      args = (x,stencil_size,
              np.array([1.0,1.0]),
              np.array([[2,0],[0,2]]), 
              reg_basis,reg_poly_order,
              vert,smp)
      yield args 

  # make submatrices for time smoothing for each station
  def time_args_maker():
    for xi in x:
      vert,smp = time_cuts.get_vert_smp(xi) 
      # note that stencil size and polynomial order are hard coded at 
      # 3 and 2
      args = (t[:,None],3,
              np.array([1.0]),
              np.array([[2]]), 
              reg_basis,2,
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

  Nx = len(x)
  Nt = len(t)

  ## combine submatrices into the master matrix
  ###################################################################
  wrapped_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
  
  rows = np.zeros((3*Nt,Nx))
  cols = np.zeros((3*Nt,Nx))
  vals = np.zeros((3*Nt,Nx))
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

  rows = np.zeros((stencil_size*Nx,Nt))
  cols = np.zeros((stencil_size*Nx,Nt))
  vals = np.zeros((stencil_size*Nx,Nt))
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
  # do not store the zeros as values
  Lt_out.eliminate_zeros()
  Lx_out.eliminate_zeros()

  return Lt_out,Lx_out


def _system_matrix(Nt,Nx):
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
  return G


def network_smoother(u,t,x,
                     sigma=None,
                     stencil_size=5,
                     stencil_space_cuts=None,
                     stencil_time_cuts=None,
                     reg_basis=rbf.basis.phs3,
                     reg_poly_order=1,
                     reg_time_parameter=None,
                     reg_space_parameter=None,
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
                     bs_itr=10,
                     procs=None):

  # check for duplicate stations 
  _identify_duplicate_stations(x)

  if cv_space_bounds is None:
    cv_space_bounds = [-4.0,4.0]
  if cv_time_bounds is None:
    cv_time_bounds = [-4.0,4.0]
 
  u = np.asarray(u)

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

  u_flat = u.flatten()
  sigma_flat = sigma.flatten()

  logger.info('building regularization matrix...')
  Lt,Lx = _reg_matrices(t,x,
                        stencil_size=stencil_size,
                        reg_basis=reg_basis,
                        reg_poly_order=reg_poly_order, 
                        time_cuts=stencil_time_cuts,
                        space_cuts=stencil_space_cuts,
                        procs=procs)
  logger.info('done')

  logger.info('building system matrix...')
  G = _system_matrix(Nt,Nx)
  logger.info('done')

  # weigh G and u by the inverse of data uncertainty. this creates 
  # duplicates but G should still be small
  W = scipy.sparse.diags(1.0/sigma_flat,0)
  G = W.dot(G)
  u_flat = W.dot(u_flat)

  # clean up any zero entries
  G.eliminate_zeros()

  # make cross validation testing sets if necessary. the testing sets 
  # are split up by station
  if (reg_time_parameter is None) | (reg_space_parameter is None):
    cv_fold = min(cv_fold,Nx)
    testing_x_sets = modest.cv.chunkify(range(Nx),cv_fold) 
    data_indices = np.arange(Nt*Nx).reshape((Nt,Nx))
    testing_sets = []
    for tx in testing_x_sets:
      testing_sets += [data_indices[:,tx].flatten()]
  
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

  # zero the initial displacements
  u_pred[0,:] = 0.0
  sigma_u_pred[0,:] = 0.0

  return u_pred,sigma_u_pred


