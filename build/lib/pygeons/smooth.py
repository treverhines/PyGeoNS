#!/usr/bin/env python
import numpy as np
import modest.cv
import rbf.smooth
import modest
import modest.solvers
import scipy.sparse
import logging
logger = logging.getLogger(__name__)

_SOLVER_DICT = {'spsolve':modest.solvers.sparse_reg_ds,
                'lgmres':modest.solvers.sparse_reg_lgmres,
                'lsqr':modest.solvers.sparse_reg_lsqr,
                'petsc':modest.solvers.sparse_reg_petsc}

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


def _bootstrap_uncertainty(G,L,solver='spsolve',itr=10,**kwargs):
  ''' 
  estimates the uncertainty for the solution to the regularized linear 
  system.  Bootstrapping is necessary because computing the model 
  covariance matrix is too expensive.  It is assumed that G is already 
  weighted by the data uncertainty
  '''
  soln = _RunningVariance()
  for i in range(itr):
    d = np.random.normal(0.0,1.0,G.shape[0])
    solni = _SOLVER_DICT[solver](G,L,d,**kwargs)
    soln.add(solni)
    logger.info('finished bootstrap iteration %s of %s' % (i+1,itr))

  return np.sqrt(soln.get_variance())


def network_smooth(u,t,x,sigma=None,
                   stencil_size=5,connectivity=None,order=1,
                   t_damping=None,x_damping=None,solver='spsolve',
                   basis=rbf.basis.phs3,x_vert=None,x_smp=None,
                   t_vert=None,t_smp=None,cv_itr=100,bs_itr=100,plot=False,
                   x_log_bounds=None,t_log_bounds=None,fold=10,
                   **kwargs):


  if x_log_bounds is None:
    x_log_bounds = [-4.0,4.0]
  if t_log_bounds is None:
    t_log_bounds = [-4.0,4.0]
 
  u = np.asarray(u)

  Nx = x.shape[0]
  Nt = t.shape[0]

  if u.shape != (Nt,Nx):
    raise TypeError('u must have shape (Nt,Nx)')

  if sigma is None:
    sigma = np.ones((Nt,Nx))
  
  sigma = np.asarray(sigma)
  if sigma.shape != (Nt,Nx):
    raise TypeError('sigma must have shape (Nt,Nx)')

  u_flat = u.flatten()
  sigma_flat = sigma.flatten()

  # form space smoothing matrix
  Lx = rbf.smooth.smoothing_matrix(x,stencil_size=stencil_size,
                                   connectivity=connectivity,
                                   order=order,basis=basis,
                                   vert=x_vert,smp=x_smp)
  # this produces the traditional finite difference matrix for a 
  # second derivative
  Lt = rbf.smooth.smoothing_matrix(t[:,None],stencil_size=5,order='max',
                                   basis=basis,vert=t_vert,smp=t_smp)
  modest.tic('building')
  # the solution for the first timestep is defined to be zero and so 
  # we do not need the first column
  Lt = Lt[:,1:]

  Lt,Lx = rbf.smooth.grid_smoothing_matrices(Lt,Lx)

  # I will be estimating baseline displacement for each station
  # which have no regularization constraints.  
  ext = scipy.sparse.csr_matrix((Lt.shape[0],Nx))
  Lt = scipy.sparse.hstack((ext,Lt))
  Lt = Lt.tocsr()

  ext = scipy.sparse.csr_matrix((Lx.shape[0],Nx))
  Lx = scipy.sparse.hstack((ext,Lx))
  Lx = Lx.tocsr()

  # build observation matrix
  G = scipy.sparse.eye(Nx*Nt)
  G = G.tocsr()

  # chop off the first Nx columns to make room for the baseline 
  # conditions
  G = G[:,Nx:]

  # add baseline elements
  Bt = scipy.sparse.csr_matrix(np.ones((Nt,1)))
  Bx = scipy.sparse.csr_matrix((0,Nx))
  Bt,Bx = rbf.smooth.grid_smoothing_matrices(Bt,Bx)
  G = scipy.sparse.hstack((Bt,G))
  G = G.tocsr()

  # weigh G and u by the inverse of data uncertainty. this creates 
  # duplicates but G should still be small
  W = scipy.sparse.diags(1.0/sigma_flat,0)
  G = W.dot(G)
  u_flat = W.dot(u_flat)

  # clean up any zero entries
  G.eliminate_zeros()

  modest.toc('building')
  # estimate damping parameters
  if (t_damping is None) & (x_damping is None):
    logger.info(
      'damping parameters were not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[t_log_bounds,x_log_bounds],solver=solver,**kwargs)
    t_damping = out[0][0] 
    x_damping = out[0][1] 
    
  elif t_damping is None:
    logger.info(
      'time damping parameter was not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[t_log_bounds,[np.log10(x_damping)-1e-4,np.log10(x_damping)+1e-4]],
            solver=solver,**kwargs)
    t_damping = out[0][0]

  elif x_damping is None:
    logger.info(
      'spatial damping parameter was not specified and will now be '
      'estimated with cross validation')
    out = modest.cv.optimal_damping_parameters(
            G,[Lt,Lx],u_flat,itr=cv_itr,fold=fold,plot=plot,
            log_bounds=[[np.log10(t_damping)-1e-4,np.log10(t_damping)+1e-4],x_log_bounds],
            solver=solver,**kwargs)
    x_damping = out[0][1]


  # this makes matrix copies
  L = scipy.sparse.vstack((t_damping*Lt,x_damping*Lx))

  logger.info('solving for predicted displacements ...')
  u_pred = _SOLVER_DICT[solver](G,L,u_flat,**kwargs)
  logger.info('finished')

  logger.info('bootstrapping uncertainty ...')
  sigma_u_pred = _bootstrap_uncertainty(G,L,solver=solver,itr=bs_itr,**kwargs)
  logger.info('finished')

  u_pred = u_pred.reshape((Nt,Nx))
  sigma_u_pred = sigma_u_pred.reshape((Nt,Nx))

  # zero the initial displacements
  u_pred[0,:] = 0.0
  sigma_u_pred[0,:] = 0.0

  return u_pred,sigma_u_pred


