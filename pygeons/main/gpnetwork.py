''' 
module of functions that construct *GaussianProcess* instances. 
'''
import numpy as np
import rbf.basis
import rbf.poly
from rbf import gauss
from pygeons.main.gptools import set_units,kernel_product
from pygeons.main import gpstation

# 2D GaussianProcess constructors
#####################################################################
def p0(sigma):
  ''' 
  Spatially constant basis function
  '''
  def basis(x,diff):
    powers = np.array([[0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  mu = np.full((1,),0.0)
  sigma = np.full((1,),sigma)
  return gauss.gpbfc(basis,mu,sigma,dim=2)  


def se(sigma,cls):
  ''' 
  Squared exponential covariance function
  '''
  return gauss.gpse((0.0,sigma**2,cls),dim=2)


def exp(sigma,cls):
  ''' 
  Exponential covariance function
  '''
  return gauss.gpexp((0.0,sigma**2,cls),dim=2)


def mat32(sigma,cls):
  ''' 
  Matern covariance function with nu=3/2
  '''
  return gauss.gpiso(rbf.basis.mat32,(0.0,sigma**2,cls),dim=2)


def mat52_2d(sigma,cls):
  ''' 
  Matern space covariance function with nu=5/2
  '''
  return gauss.gpiso(rbf.basis.mat52,(0.0,sigma**2,cls),dim=2)


# 3D GaussianProcess constructors
#####################################################################
@set_units([])
def p00():
  ''' 
  Constant in time and space basis function (i.e. {1})

  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[0,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  

@set_units([])
def p01():
  ''' 
  Constant in time and linear in space basis functions (i.e. {x,y})

  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[0,1,0],[0,0,1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p10():
  ''' 
  Linear in time and constant in space basis function (i.e. {t})

  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[1,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p11():
  ''' 
  Linear in time and linear in space basis functions (i.e. {tx,ty}) 

  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[1,1,0],[1,0,1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out

  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p20():
  ''' 
  Quadratic in time and constant in space basis functions (i.e. {t^2})
  
  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[2,0,0]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  return gauss.gpbfci(basis,dim=3)  


@set_units([])
def p21():
  ''' 
  Quadratic in time and linear in space basis functions (i.e. {t^2x,t^2y})

  Parameters
  ----------
  None  
  '''
  def basis(x,diff):
    powers = np.array([[2,1,0],[2,0,1]])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  return gauss.gpbfci(basis,dim=3)  
  

@set_units(['mm','yr','km'])
def se_se(sigma,cts,cls):
  ''' 
  SE in time and space covariance function 
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.se(sigma,cts,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def exp_se(sigma,cts,cls):
  ''' 
  EXP in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.exp(sigma,cts,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr'])
def exp_p0(sigma,cts):
  ''' 
  EXP in time and constant in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  '''
  tgp = gpstation.exp(sigma,cts,convert=False)
  sgp = p0(1.0)
  return kernel_product(tgp,sgp)


@set_units(['mm/yr^0.5','1/yr','km'])
def fogm_se(sigma,fc,cls):
  ''' 
  FOGM in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm/yr^0.5] : Standard deviation of forcing term
  fc [1/yr] : Cutoff frequency
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.fogm(sigma,fc,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm/yr^0.5','mjd','km'])
def bm_se(sigma,t0,cls):
  ''' 
  BM in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm/yr^0.5]: Standard deviation of forcing term
  t0 [mjd] : Start time
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.bm(sigma,t0,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm/yr^1.5','mjd','km'])
def ibm_se(sigma,t0,cls):
  ''' 
  IBM in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm/yr^1.5] : Standard deviation of forcing term
  fc [mjd] : Cutoff frequency
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.ibm(sigma,t0,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def mat32_se(sigma,cts,cls):
  ''' 
  MAT32 in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.mat32(sigma,cts,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def mat52_se(sigma,cts,cls):
  ''' 
  MAT52 in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.mat52(sigma,cts,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','km'])
def per_se(sigma,cls):
  ''' 
  PER in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of coefficients
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.per(sigma,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','mjd','km'])
def step_se(sigma,t0,cls):
  ''' 
  STEP in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of step size
  t0 [mjd] : Step time
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.step(sigma,t0,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


#CONSTRUCTORS = {'p00':p00,
#                'p01':p01,
#                'p10':p10,
#                'p11':p11,
#                'p20':p20,
#                'p21':p21,
#                'per-se':per_se,
#                'step-se':step_se,
#                'bm-se':bm_se,
#                'ibm-se':ibm_se,
#                'fogm-se':fogm_se,
#                'mat32-se':mat32_se,
#                'mat52-se':mat52_se,
#                'se-se':se_se,
#                'exp-se':exp_se}

# trim down constructors to the ones that are useful for strain
# analysis
CONSTRUCTORS = {'p00':p00,
                'p01':p01,
                'p10':p10,
                'p11':p11,
                'per-se':per_se,
                'bm-se':bm_se,
                'ibm-se':ibm_se,
                'fogm-se':fogm_se,
                'se-se':se_se,
                'exp-se':exp_se,
                'exp-p0':exp_p0,
                'mat32-se':mat32_se,
                'mat52-se':mat52_se}
