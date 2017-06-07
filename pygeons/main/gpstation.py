''' 
Module of station Gaussian process constructors.
'''
import numpy as np
import rbf.basis
import rbf.poly
from rbf import gauss
from pygeons.main.gptools import set_units
                               

@set_units([])
def p0():
  ''' 
  Constant basis function  
  
  Parameters
  ----------
  None
  '''
  def basis(x,diff):
    # use 2000-01-01 as the reference time which is 51544 in MJD
    t0 = np.array([51544.0])
    powers = np.array([[0]])
    out = rbf.poly.mvmonos(x - t0,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units([])
def p1():
  ''' 
  Linear basis function

  Parameters
  ----------
  None
  '''
  def basis(x,diff):
    # use 2000-01-01 as the reference time which is 51544 in MJD
    t0 = np.array([51544.0])
    powers = np.array([[1]])
    out = rbf.poly.mvmonos(x - t0,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units([])
def const():
  ''' 
  Constant in time basis function  
  
  Parameters
  ----------
  None
  '''
  def basis(x,diff):
    # use 2000-01-01 as the reference time which is 51544 in MJD
    t0 = np.array([51544.0])
    powers = np.array([[0]])
    out = rbf.poly.mvmonos(x - t0,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units([])
def linear():
  ''' 
  Constant and linear in time basis functions
  
  Parameters
  ----------
  None
  '''
  def basis(x,diff):
    # use 2000-01-01 as the reference time which is 51544 in MJD
    t0 = np.array([51544.0])
    powers = np.array([[0],[1]])
    out = rbf.poly.mvmonos(x - t0,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units([])
def per():
  ''' 
  Annual and semiannual sinusoid basis functions 
  
  Parameters
  ----------
  None
  '''
  def basis(x):
    # no derivatives because im lazy
    out = np.array([np.sin(2*np.pi*x[:,0]/365.25),
                    np.cos(2*np.pi*x[:,0]/365.25),
                    np.sin(4*np.pi*x[:,0]/365.25),
                    np.cos(4*np.pi*x[:,0]/365.25)]).T
    return out

  return gauss.gpbfci(basis,dim=1)


@set_units(['mjd'])
def step(t0):
  ''' 
  Heaviside step function
  
  Parameters
  ----------
  t0 [mjd] : Time of the step
  '''
  def basis(x,diff):
    if diff == (0,):
      out = (x[:,0] >= t0).astype(float)
    else:
      # derivative is zero everywhere (ignore the step at t0)
      out = np.zeros(x.shape[0],dtype=float)  
    
    # turn into an (N,1) array
    out = out[:,None]
    return out

  return gauss.gpbfci(basis,dim=1)


@set_units(['mjd'])
def ramp(t0):
  ''' 
  Ramp function 
  
  Parameters
  ----------
  t0 [mjd] : Time of the ramp
  '''
  def basis(x,diff):
    if diff == (0,):
      out = (x[:,0] - t0)*((x[:,0] >= t0).astype(float))
    elif diff == (1,):
      out = (x[:,0] >= t0).astype(float)
    else:
      # derivative is zero everywhere (ignore the step at t0)
      out = np.zeros(x.shape[0],dtype=float)  
    
    # turn into an (N,1) array
    out = out[:,None]
    return out

  return gauss.gpbfci(basis,dim=1)


@set_units(['mm','yr'])
def mat32(sigma,cts):
  ''' 
  Matern covariance function with nu=3/2
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.mat32,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def mat52(sigma,cts):
  ''' 
  Matern covariance function with nu=5/2
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.mat52,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def wen12(sigma,cts):
  ''' 
  Wendland 1-D C4 covariance function. 
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.wen12,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def spwen12(sigma,cts):
  ''' 
  Wendland 1-D C4 covariance function. 
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.spwen12,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def wen30(sigma,cts):
  ''' 
  Wendland 3-D C0 covariance function. 
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.wen30,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def spwen30(sigma,cts):
  ''' 
  Wendland 3-D C0 covariance function. 
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpiso(rbf.basis.spwen30,(0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def se(sigma,cts):
  ''' 
  Squared exponential covariance function
  
  cov(t,t') = sigma^2 * exp(-|t - t'|^2/(2*cts^2))

  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpse((0.0,sigma**2,cts),dim=1)


@set_units(['mm','yr'])
def exp(sigma,cts):
  ''' 
  Exponential covariance function
  
  cov(t,t') = sigma^2 * exp(-|t - t'|/cts)

  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  return gauss.gpexp((0.0,sigma**2,cts),dim=1)


@set_units(['mm/yr^0.5','yr^-1'])
def fogm(sigma,w):
  ''' 
  First-order Gauss Markov process
    
  cov(t,t') = sigma^2/(2*w) * exp(-w*|t - t'|)  

  Parameters
  ----------
  sigma [mm/yr^0.5] : Standard deviation of the forcing term
  w [yr^-1] : Cutoff angular frequency 
  '''
  coeff = sigma**2/(2*w)
  cts   = 1.0/w
  return gauss.gpexp((0.0,coeff,cts),dim=1)


@set_units(['mm/yr^0.5','mjd'])
def bm(sigma,t0):
  ''' 
  Brownian motion 
  
  cov(t,t') = sigma^2 * min(t - t0,t' - t0) 

  Parameters
  ----------
  sigma [mm/yr^0.5] : Standard deviation of the forcing term
  t0 [mjd] : Start time
  '''
  def mean(x):
    out = np.zeros(x.shape[0])  
    return out 

  def cov(x1,x2):
    x1,x2 = np.meshgrid(x1[:,0]-t0,x2[:,0]-t0,indexing='ij')
    x1[x1<0.0] = 0.0
    x2[x2<0.0] = 0.0
    out = np.min([x1,x2],axis=0)
    return sigma**2*out
  
  return gauss.GaussianProcess(mean,cov,dim=1)  


@set_units(['mm/yr^1.5','mjd'])
def ibm(sigma,t0):
  ''' 
  Integrated Brownian motion 
  
  x = t - t0  
  cov(t,t') = sigma^2 * min(x,x')^2 * (max(x,x') - min(x,x')/3) / 2 

  Parameters
  ----------
  sigma [mm/yr^1.5] : Standard deviation of the forcing term
  t0 [mjd] : Start time
  '''
  def mean(x,diff):
    '''mean function which is zero for all derivatives'''
    out = np.zeros(x.shape[0])
    return out
  
  def cov(x1,x2,diff1,diff2):
    '''covariance function and its derivatives'''
    x1,x2 = np.meshgrid(x1[:,0]-t0,x2[:,0]-t0,indexing='ij')
    x1[x1<0.0] = 0.0
    x2[x2<0.0] = 0.0
    if (diff1 == (0,)) & (diff2 == (0,)):
      # integrated brownian motion
      out = (0.5*np.min([x1,x2],axis=0)**2*
             (np.max([x1,x2],axis=0) -
              np.min([x1,x2],axis=0)/3.0))
  
    elif (diff1 == (1,)) & (diff2 == (1,)):
      # brownian motion
      out = np.min([x1,x2],axis=0)
  
    elif (diff1 == (1,)) & (diff2 == (0,)):
      # derivative w.r.t x1
      out = np.zeros_like(x1)
      idx1 = x1 >= x2
      idx2 = x1 <  x2
      out[idx1] = 0.5*x2[idx1]**2
      out[idx2] = x1[idx2]*x2[idx2] - 0.5*x1[idx2]**2
  
    elif (diff1 == (0,)) & (diff2 == (1,)):
      # derivative w.r.t x2
      out = np.zeros_like(x1)
      idx1 = x2 >= x1
      idx2 = x2 <  x1
      out[idx1] = 0.5*x1[idx1]**2
      out[idx2] = x2[idx2]*x1[idx2] - 0.5*x2[idx2]**2
  
    else:
      raise ValueError(
        'The *GaussianProcess* is not sufficiently differentiable')
  
    return sigma**2*out

  return gauss.GaussianProcess(mean,cov,dim=1)  


CONSTRUCTORS = {'p0':p0, 
                'p1':p1, 
                'const':const,
                'linear':linear,
                'per':per, 
                'step':step,
                'ramp':ramp,
                'bm':bm,
                'ibm':ibm,
                'fogm':fogm,
                'mat32':mat32,
                'mat52':mat52,
                'wen12':wen12,
                'spwen12':spwen12,
                'wen30':wen30,
                'spwen30':spwen30,
                'se':se,
                'exp':exp}
