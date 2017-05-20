''' 
module of network Gaussian process constructors.
'''
import rbf.basis
import rbf.poly
from rbf import gauss
from pygeons.main.gptools import set_units,kernel_product
from pygeons.main import gpstation

# 2D GaussianProcess constructors
#####################################################################
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


def mat52(sigma,cls):
  ''' 
  Matern space covariance function with nu=5/2
  '''
  return gauss.gpiso(rbf.basis.mat52,(0.0,sigma**2,cls),dim=2)

def wen32(sigma,cls):
  ''' 
  Wendland space covariance function
  '''
  return gauss.gpiso(rbf.basis.wen32,(0.0,sigma**2,cls),dim=2)

def spwen32(sigma,cls):
  ''' 
  Sparse Wendland space covariance function 
  '''
  return gauss.gpiso(rbf.basis.spwen32,(0.0,sigma**2,cls),dim=2)


# 3D GaussianProcess constructors
#####################################################################
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


@set_units(['mm','yr','km'])
def wen12_se(sigma,cts,cls):
  ''' 
  WEN11 in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.wen12(sigma,cts,convert=False)
  sgp = se(1.0,cls)
  return kernel_product(tgp,sgp)


@set_units(['mm','yr','km'])
def spwen12_se(sigma,cts,cls):
  ''' 
  WEN11 in time and SE in space covariance function
  
  Parameters
  ----------
  sigma [mm] : Standard deviation of displacements
  cts [yr] : Characteristic time-scale
  cls [km] : Characteristic length-scale
  '''
  tgp = gpstation.spwen12(sigma,cts,convert=False)
  sgp = se(1.0,cls)
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

# trim down constructors to the ones that are useful for strain
# analysis
CONSTRUCTORS = {'bm-se':bm_se,
                'ibm-se':ibm_se,
                'fogm-se':fogm_se,
                'se-se':se_se,
                'exp-se':exp_se,
                'mat32-se':mat32_se,
                'mat52-se':mat52_se,
                'wen12-se':wen12_se,
                'spwen12-se':spwen12_se}
