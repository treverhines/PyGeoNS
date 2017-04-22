''' 
Contains a Gaussian process regression function that has been 
specialized for PyGeoNS
'''
import numpy as np
import logging
import rbf.gauss
from pygeons.main.gprocs import gpcomp
logger = logging.getLogger(__name__)


def _station_sigma_and_p(gp,time,mask):
  ''' 
  Build the covariance matrix and basis vectors describing noise for
  all stations. The covariance and basis functions will only be
  evauluated at unmasked data.
  '''
  # stations that *will* have basis vectors
  sigma_i = gp.covariance(time,time)
  p_i = gp.basis(time)
  _,Nx = mask.shape
  _,Np = p_i.shape
  
  sigma = np.zeros((np.sum(~mask),np.sum(~mask)))
  p = np.zeros((np.sum(~mask),Np,Nx))

  r,c = np.nonzero(~mask)
  for i in range(Nx):
    # good luck trying to figure out what the fuck im doing here
    sigma[c==i,c==i] = sigma_i[r[c==i],r[c==i]]
    p[c==i,:,i] = p_i[r[c==i]]

  p = p.reshape((np.sum(~mask),Np*Nx))
  # remove singluar values from p
  u,s,_ = np.linalg.svd(p,full_matrices=False)
  p = u[:,s>1e-10]
  logger.debug('Removed %s singular values from the station basis vectors' 
               % sum(~(s>1e-10)))
  return sigma,p


def strain(t,x,d,sd,
           prior_model=('se-se',),
           prior_params=(1.0,0.05,50.0),
           noise_model=('null',),
           noise_params=(),
           station_noise_model=('p0',),
           station_noise_params=(),
           out_t=None,
           out_x=None,
           tol=4.0):
  ''' 
  Computes deformation gradients from displacement data.
  
  Parameters
  ----------
  t : (Nt,1) array
  x : (Nx,2) array
  d : (Nt,Nx) array
  sd : (Nt,Nx) array
  prior_model : str array
  prior_params : float array
  noise_model : str array
  noise_params : float array
  station_noise_model : str array
  station_noise_params : float array
  out_t : (Mt,) array, optional
  out_x : (Mx,2) array, optional
  tol : float, optional
    
  Returns
  -------
  de: (Nt,Nx) array
    edited data
  sde: (Nt,Nx) array
    edited data uncertainty
  fit: (Nt,Nx) float array
  dx: (Mt,Mx) array
  sdx: (Mt,Mx) array
  dy: (Mt,Mx) array
  sdy: (Mt,Mx) array
  '''  
  t = np.asarray(t,dtype=float)
  x = np.asarray(x,dtype=float)
  de = np.array(d,dtype=float,copy=True)
  sde = np.array(sd,dtype=float,copy=True)
  Nt,Nx = t.shape[0],x.shape[0]
  # allocate array indicating which data have been removed
  if out_t is None:
    out_t = t

  if out_x is None:
    out_x = x

  prior_gp = gpcomp(prior_model,prior_params)
  noise_gp = gpcomp(noise_model,noise_params)    
  sta_gp   = gpcomp(station_noise_model,station_noise_params)

  t_grid,x0_grid = np.meshgrid(t,x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(t,x[:,1],indexing='ij')  
  # flat observation times and positions
  z = np.array([t_grid.ravel(),
                x0_grid.ravel(),
                x1_grid.ravel()]).T

  t_grid,x0_grid = np.meshgrid(out_t,out_x[:,0],indexing='ij')  
  t_grid,x1_grid = np.meshgrid(out_t,out_x[:,1],indexing='ij')  
  # flat observation times and positions
  out_z = np.array([t_grid.ravel(),
                    x0_grid.ravel(),
                    x1_grid.ravel()]).T

  # Build covariance and basis vectors for the combined process. Do
  # not evaluated at masked points
  mask = np.isinf(sde)
  full_sigma,full_p = _station_sigma_and_p(sta_gp,t,mask)
  full_sigma += noise_gp.covariance(z[~mask.ravel()],z[~mask.ravel()])
  full_sigma += prior_gp.covariance(z[~mask.ravel()],z[~mask.ravel()])
  full_p = np.hstack((full_p,noise_gp.basis(z[~mask.ravel()])))
  full_p = np.hstack((full_p,prior_gp.basis(z[~mask.ravel()])))
  full_mu = np.zeros(np.sum(~mask))  
  # returns the indices of outliers 
  outliers,fitf = rbf.gauss.outliers(de[~mask],sde[~mask],
                                     mu=full_mu,sigma=full_sigma,p=full_p,
                                     tol=tol,return_fit=True)
  # dereference full_* since we will not be using them anymore
  del full_sigma,full_p,full_mu
  
  # mask the outliers in *de* and *sde*
  r,c = np.nonzero(~mask)
  de[r[outliers],c[outliers]] = np.nan
  sde[r[outliers],c[outliers]] = np.inf
  
  # best fit combination of signal and noise to the observations
  fit = np.full((Nt,Nx),np.nan)
  fit[~mask] = fitf
  
  # update the mask to include outliers
  mask = np.isinf(sde)
  # rebuild noise covariance and basis vectors
  noise_sigma,noise_p = _station_sigma_and_p(sta_gp,t,mask)
  noise_sigma += noise_gp.covariance(z[~mask.ravel()],z[~mask.ravel()])
  noise_p = np.hstack((noise_p,noise_gp.basis(z[~mask.ravel()])))
  rbf.gauss._diag_add(noise_sigma,sde[~mask]**2)
  
  # condition the prior with the data
  post_gp = prior_gp.condition(z[~mask.ravel()],de[~mask],sigma=noise_sigma,p=noise_p)
  dx_gp = post_gp.differentiate((1,1,0)) # x derivative of velocity
  dy_gp = post_gp.differentiate((1,0,1)) # y derivative of velocity

  dx,sdx = dx_gp.meansd(out_z)
  dx = dx.reshape((out_t.shape[0],out_x.shape[0]))
  sdx = sdx.reshape((out_t.shape[0],out_x.shape[0]))

  dy,sdy = dy_gp.meansd(out_z)
  dy = dy.reshape((out_t.shape[0],out_x.shape[0]))
  sdy = sdy.reshape((out_t.shape[0],out_x.shape[0]))

  out  = (de,sde,fit,dx,sdx,dy,sdy)
  return out
