#!/usr/bin/env python
import numpy as np
import modest.cv
import modest
import modest.solvers
import scipy.sparse
import logging
from scipy.spatial import cKDTree
import pygeons.cuts
import modest.mp
import pygeons.diff

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


def network_smoother(u,t,x,
                     sigma=None,
                     diff_specs=None,
                     penalties=None, 
                     solve_ksp='lgmres',
                     solve_pc='icc',
                     solve_max_itr=1000,
                     solve_atol=1e-6, 
                     solve_rtol=1e-8, 
                     solve_view=False, 
                     cv_itr=100,
                     cv_bounds=None, 
                     cv_plot=False,
                     cv_fold=10,
                     cv_chunk='both',
                     perts=10,
                     procs=None,
                     baseline=False):

  u = np.asarray(u)
  t = np.asarray(t)
  x = np.asarray(x)

  Nx = x.shape[0]
  Nt = t.shape[0]

  if diff_specs is None:
    diff_specs = [pygeons.diff.ACCELERATION,
                  pygeons.diff.VELOCITY_LAPLACIAN]

  R = len(diff_specs)

  if cv_bounds is None:
    cv_bounds = [[-4.0,4.0] for i in range(R)]
 
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

  u_flat = u.ravel()
  sigma_flat = sigma.ravel()

  logger.info('building regularization matrix...')
  reg_matrices = [pygeons.diff._diff_matrix(t,x,d) for d in diff_specs]
  logger.info('done')

  # system matrix is the identity matrix scaled by data weight
  Gdata = 1.0/sigma_flat
  Grow = range(Nt*Nx)
  Gcol = range(Nt*Nx)
  Gsize = (Nt*Nx,Nt*Nx)
  G = scipy.sparse.csr_matrix((Gdata,(Grow,Gcol)),Gsize)
  
  # weigh u by the inverse of data uncertainty.
  u_flat = u_flat/sigma_flat

  # make cross validation testing sets if necessary. the testing sets 
  # are split up by station
  if penalties is None:
    logger.info(
      'penalty parameters were not specified and will now be '
      'estimated with cross validation')

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


    out = modest.cv.optimal_damping_parameters(
            G,reg_matrices,u_flat,itr=cv_itr,fold=testing_sets,
            plot=cv_plot,log_bounds=cv_bounds,
            solver='petsc',ksp=solve_ksp,pc=solve_pc,
            maxiter=solve_max_itr,view=solve_view,atol=solve_atol,
            rtol=solve_rtol,procs=procs)

    penalties = out[0]

  # this makes matrix copies
  L = scipy.sparse.vstack(p*r for p,r in zip(penalties,reg_matrices))

  logger.info('solving for predicted displacements...')
  u_pred = modest.sparse_reg_petsc(
             G,L,u_flat,
             ksp=solve_ksp,pc=solve_pc,
             maxiter=solve_max_itr,view=solve_view,
             atol=solve_atol,rtol=solve_rtol)
  logger.info('done')

  logger.info('computing perturbed predicted displacements...')
  u_pert = np.zeros((perts,G.shape[0]))
  for i in range(perts):
    d = np.random.normal(0.0,1.0,G.shape[0])
    u_pert[i,:] = modest.sparse_reg_petsc(G,L,d,
                    ksp=solve_ksp,pc=solve_pc,
                    maxiter=solve_max_itr,view=solve_view,
                    atol=solve_atol,rtol=solve_rtol)
    u_pert[i,:] += u_pred

  logger.info('done')

  u_pred = u_pred.reshape((Nt,Nx))
  u_pert = u_pert.reshape((perts,Nt,Nx))

  return u_pred,u_pert


