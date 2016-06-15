#!/usr/bin/env python
import numpy as np
import modest.cv
import modest
import modest.solvers
import scipy.sparse
import logging
import pygeons.cuts
import modest.mp
import pygeons.diff

logger = logging.getLogger(__name__)

def network_smoother(u,t,x,
                     sigma=None,
                     diff_specs=None,
                     penalties=None, 
                     use_umfpack=True,
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
  reg_matrices = [pygeons.diff.diff_matrix(t,x,d,procs=procs) for d in diff_specs]
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
            solver='spsolve',use_umfpack=use_umfpack)

    penalties = out[0]

  # this makes matrix copies
  L = scipy.sparse.vstack(p*r for p,r in zip(penalties,reg_matrices))
  L.eliminate_zeros()
  
  logger.info('solving for predicted displacements...')
  u_pred = modest.sparse_reg_dsolve(G,L,u_flat,use_umfpack=use_umfpack)
  logger.info('done')

  logger.info('computing perturbed predicted displacements...')
  u_pert = np.zeros((perts,G.shape[0]))
  # perturbed displacements will be computed in parallel and so this 
  # needs to be turned into a mappable function
  def calculate_pert(args):
    G = args[0]
    L = args[1]
    d = args[2]
    use_umfpack=args[3]
    return modest.sparse_reg_dsolve(G,L,d,use_umfpack=use_umfpack)

  # generator for arguments that will be passed to calculate_pert
  args = ((G,L,np.random.normal(0.0,1.0,G.shape[0]),use_umfpack)
           for i in range(perts))
  u_pert = modest.mp.parmap(calculate_pert,args,workers=procs)
  u_pert = np.reshape(u_pert,(perts,(Nt*Nx)))
  u_pert += u_pred[None,:]

  logger.info('done')

  u_pred = u_pred.reshape((Nt,Nx))
  u_pert = u_pert.reshape((perts,Nt,Nx))

  return u_pred,u_pert


