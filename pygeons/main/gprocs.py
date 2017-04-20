''' 
module of functions that construct *GaussianProcess* instances. 
'''
import numpy as np
import rbf.basis
import rbf.poly
from rbf import gauss
from pygeons.units import unit_conversion as conv


def set_units(units):
  ''' 
  Wrapper for specifying the hyperparameter units for each
  GaussianProcess constructor. When a wrapped constructor is called,
  the hyperparameters are converted to be in terms of *m* and *day*.
  The constructor is also given the key word argument *convert*, which
  can be set to False if no conversion is desired.
  '''
  def decorator(fin):
    def fout(*args,**kwargs):
      convert = kwargs.pop('convert',False)
      if convert:
        args = [a*conv(u,time='day',space='m') for a,u in zip(args,units)]
        
      return fin(*args)

    fout.units = units
    fout.n = gauss._get_arg_count(fin)
    if fout.n != len(fout.units):
      raise ValueError(
        'the number of arguments must be equal to the number of unit '
        'specifications') 
    
    return fout  

  return decorator  


@set_units([])
def null():
  '''Null GaussianProcess'''
  return gauss.GaussianProcess(gauss._zero_mean,
                               gauss._zero_covariance,
                               basis=gauss._empty_basis)
                               
# 1D GaussianProcess constructors
#####################################################################
@set_units([])
def p0_1d():
  ''' 
  Constant in time monomial
  '''
  def basis(x,diff):
    powers = np.array([[0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units([])
def p1_1d():
  ''' 
  Linear in time monomial
  '''
  def basis(x,diff):
    powers = np.array([[1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=1)  


@set_units(['mm'])
def per_1d(a):
  ''' 
  Annual and semiannual sinusoids. Coefficients have std. dev. *a*
  '''
  def basis(x):
    # no derivatives because im lazy
    out = np.array([np.sin(2*np.pi*x[:,0]/365.25),
                    np.cos(2*np.pi*x[:,0]/365.25),
                    np.sin(4*np.pi*x[:,0]/365.25),
                    np.cos(4*np.pi*x[:,0]/365.25)]).T
    return out

  mu = np.zeros(4)
  sigma = np.full(4,a)
  return gauss.gpbfc(basis,mu,sigma,dim=1)


@set_units([])
def peri_1d():
  ''' 
  Annual and semiannual sinusoids. Coefficients are unconstrainted
  '''
  def basis(x):
    # no derivatives because im lazy
    out = np.array([np.sin(2*np.pi*x[:,0]/365.25),
                    np.cos(2*np.pi*x[:,0]/365.25),
                    np.sin(4*np.pi*x[:,0]/365.25),
                    np.cos(4*np.pi*x[:,0]/365.25)]).T
    return out

  return gauss.gpbfci(basis,dim=1)


@set_units(['mm','mjd'])
def step_1d(a,t0):
  ''' 
  Step function centered at *t0*. Step size have std. dev. *a*
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

  mu = np.zeros(1)
  sigma = np.full(1,a)
  return gauss.gpbfc(basis,mu,sigma,dim=1)


@set_units(['mjd'])
def stepi_1d(t0):
  ''' 
  Step function centered at *t0*. Step size is unconstrained
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


@set_units(['mm','mjd'])
def ramp_1d(a,t0):
  ''' 
  Ramp function centered at *t0*. Slope has std. dev. given by *a*
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

  mu = np.zeros(1)
  sigma = np.full(1,a)
  return gauss.gpbfc(basis,mu,sigma,dim=1)


@set_units(['mm','yr'])
def mat32_1d(sigma,cls):
  ''' 
  Matern time covariance function with nu=3/2
  '''
  return gauss.gpiso(rbf.basis.mat32,(0.0,sigma**2,cls),dim=1)


@set_units(['mm','yr'])
def mat52_1d(sigma,cls):
  ''' 
  Matern time covariance function with nu=5/2
  '''
  return gauss.gpiso(rbf.basis.mat52,(0.0,sigma**2,cls),dim=1)


@set_units(['mm','yr'])
def se_1d(sigma,cls):
  ''' 
  Squared exponential time covariance function 
  '''
  return gauss.gpse((0.0,sigma**2,cls),dim=1)


@set_units(['mm','yr'])
def exp_1d(sigma,cls):
  ''' 
  Exponential time covariance function 
  '''
  return gauss.gpexp((0.0,sigma**2,cls),dim=1)


@set_units(['mm/yr^0.5','yr^-1'])
def fogm_1d(s,fc):
  ''' 
  First order Gauss Markov time covariance function. The
  autocovariance function is
    
     K(t) = s^2/(4*pi*fc) * exp(-2*pi*fc*|t|)  
   
  which has the corresponding power spectrum 
  
     P(f) = s^2/(4*pi^2 * (f^2 + fc^2))
  
  *fc* can be interpretted as a cutoff frequency which marks the
  transition to a flat power spectrum and a power spectrum that decays
  with a spectral index of two. Thus, when *fc* is close to zero, the
  power spectrum resembles that of Brownian motion.
  '''
  coeff = s**2/(4*np.pi*fc)
  cls   = 1.0/(2*np.pi*fc)
  return gauss.gpexp((0.0,coeff,cls),dim=1)


@set_units(['mm/yr^0.5','mjd'])
def bm_1d(w,t0):
  ''' 
  Brownian motion 
  '''
  def mean(x):
    out = np.zeros(x.shape[0])  
    return out 

  def cov(x1,x2):
    x1,x2 = np.meshgrid(x1[:,0]-t0,x2[:,0]-t0,indexing='ij')
    x1[x1<0.0] = 0.0
    x2[x2<0.0] = 0.0
    out = np.min([x1,x2],axis=0)
    return w**2*out
  
  return gauss.GaussianProcess(mean,cov,dim=1)  


@set_units(['mm/yr^1.5','mjd'])
def ibm_1d(w,t0):
  ''' 
  Integrated Brownian motion 
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
  
    return w**2*out

  return gauss.GaussianProcess(mean,cov,dim=1)  

# 2D GaussianProcess constructors
#####################################################################
@set_units(['mm','km'])
def se_2d(sigma,cls):
  ''' 
  Squared exponential space covariance function
  '''
  return gauss.gpse((0.0,sigma**2,cls),dim=2)


@set_units(['mm','km'])
def exp_2d(sigma,cls):
  ''' 
  Exponential space covariance function
  '''
  return gauss.gpexp((0.0,sigma**2,cls),dim=2)


@set_units(['mm','km'])
def mat32_2d(sigma,cls):
  ''' 
  Matern space covariance function with nu=3/2
  '''
  return gauss.gpiso(rbf.basis.mat32,(0.0,sigma**2,cls),dim=2)


@set_units(['mm','km'])
def mat52_2d(sigma,cls):
  ''' 
  Matern space covariance function with nu=5/2
  '''
  return gauss.gpiso(rbf.basis.mat52,(0.0,sigma**2,cls),dim=2)


# 3D GaussianProcess constructors
#####################################################################
# GaussianProcesses constructors for strain calculation
def kernel_product(gp1,gp2):
  def mean(x,diff):
    return np.zeros(x.shape[0])

  def covariance(x1,x2,diff1,diff2):
    out  = gp1._covariance(x1[:,[0]],x2[:,[0]],
                           diff1[[0]],diff2[[0]])
    out *= gp2._covariance(x1[:,[1,2]],x2[:,[1,2]],
                           diff1[[1,2]],diff2[[1,2]])
    return out
  
  return gauss.GaussianProcess(mean,covariance,dim=3)  


@set_units([])
def p00_3d():
  ''' 
  Gaussian process with polynomial basis functions [(0,0,0)]
  '''
  def basis(x,diff):
    powers = np.array([[0,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p10_3d():
  ''' 
  Gaussian process with polynomial basis functions [(1,0,0)]
  '''
  def basis(x,diff):
    powers = np.array([[1,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p11_3d():
  ''' 
  Gaussian process with polynomial basis functions [(1,1,0),(1,0,1)]
  '''
  def basis(x,diff):
    powers = np.array([[1,1,0],[1,0,1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p20_3d():
  ''' 
  Gaussian process with polynomial basis functions [(2,0,0)]
  '''
  def basis(x,diff):
    powers = np.array([[2,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p21_3d():
  ''' 
  Gaussian process with polynomial basis functions [(2,1,0),(2,0,1)]
  '''
  def basis(x,diff):
    powers = np.array([[2,1,0],[2,0,1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  return gauss.gpbfci(basis,dim=3)  
  

@set_units(['mm','yr','km'])
def se_se_3d(a,b,c):
  tgp = se_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def exp_se_3d(a,b,c):
  tgp = exp_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)

@set_units(['mm*yr^-0.5','yr^-1','km'])
def fogm_se_3d(a,b,c):
  tgp = fogm_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm*yr^-0.5','mjd','km'])
def bm_se_3d(a,b,c):
  tgp = bm_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm*yr^-1.5','mjd','km'])
def ibm_se_3d(a,b,c):
  tgp = ibm_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def mat32_se_3d(a,b,c):
  tgp = mat32_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def mat52_se_3d(a,b,c):
  tgp = mat52_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','km'])
def per_se_3d(a,b):
  tgp = per_1d(a,convert=False)
  sgp = se_2d(1.0,b,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','mjd','km'])
def step_se_3d(a,b,c):
  tgp = step_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)


@set_units(['mm','mjd','km'])
def ramp_se_3d(a,b,c):
  tgp = ramp_1d(a,b,convert=False)
  sgp = se_2d(1.0,c,convert=False)
  return kernel_product(tgp,sgp)
    

# create a dictionary of all Gaussian process constructors in this
# module
CONSTRUCTORS = {'null':null,
                'p0':p0_1d, 
                'p1':p1_1d, 
                'peri':peri_1d, 
                'per':per_1d, 
                'stepi':stepi_1d,
                'step':step_1d,
                'bm':bm_1d,
                'ibm':ibm_1d,
                'fogm':fogm_1d,
                'mat32':mat32_1d,
                'mat52':mat52_1d,
                'se':se_1d,
                'exp':exp_1d,
                'p00':p00_3d,
                'p10':p10_3d,
                'p11':p11_3d,
                'p20':p20_3d,
                'p21':p21_3d,
                'per-se':per_se_3d,
                'step-se':step_se_3d,
                'bm-se':bm_se_3d,
                'ibm-se':ibm_se_3d,
                'fogm-se':fogm_se_3d,
                'mat32-se':mat32_se_3d,
                'mat52-se':mat52_se_3d,
                'se-se':se_se_3d,
                'exp-se':exp_se_3d}
  
def gpcomp(models,args):
  ''' 
  Returns a composite Gaussian process. The components are specified
  with the *model* string, and the arguments for each component are
  specied with *args*. For example,
  
  >>> gpcomp(['fogm','se'],[1.0,2.0,3.0,4.0]) 
  
  creates a GaussianProcess composed of a FOGM and an SE model. The
  first two arguments are passed to *gpfogm* and the second two
  arguments are passed to *se*
  '''
  args = list(args)
  models = list(models)
  cs = [CONSTRUCTORS[m] for m in models] # constructor for each component
  n  = sum(ci.n for ci in cs)
  if len(args) != n:
    raise ValueError(
      '%s parameters were specified for the model "%s", but it '
      'requires %s parameters.\n' %(len(args),' '.join(models),n))
  
  gp = null()
  for ci in cs:
    gp += ci(*(args.pop(0) for i in range(ci.n)))
  
  return gp
  

def get_units(models):
  ''' 
  returns the units for the GaussianProcess parameters
  '''
  models = list(models)
  cs = [CONSTRUCTORS[m] for m in models]
  units = []
  for ci in cs:
    units += ci.units
  
  return units  
