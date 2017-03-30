''' 
module of functions that generate GaussianProcesses
'''
import numpy as np
from rbf import gauss
from rbf.gauss import gppoly


def gpnull():
  '''Null GaussianProcess'''
  return gauss.GaussianProcess(gauss._zero_mean,
                               gauss._zero_covariance,
                               basis=gauss._empty_basis)

def gpseasonal(annual,semiannual):
  ''' 
  Returns a *GaussianProcess* with annual and semiannual sinusoids as
  improper basis functions. The input times should have units of years
  '''
  def basis(x):
    out = np.zeros((x.shape[0],0))
    if annual:
      # note that x is in years
      terms = np.array([np.sin(2*np.pi*x[:,0]),
                        np.cos(2*np.pi*x[:,0])]).T
      out = np.hstack((out,terms))

    if semiannual:
      terms = np.array([np.sin(4*np.pi*x[:,0]),
                        np.cos(4*np.pi*x[:,0])]).T
      out = np.hstack((out,terms))

    return out

  return gauss.gpbfci(basis,dim=1)


def gpse(sigma,cls):
  ''' 
  Returns a *GaussianProcess* with zero mean and a squared exponential
  covariance
  '''
  return gauss.gpse((0.0,sigma**2,cls))


def gpfogm(s,fc):
  ''' 
  Returns a *GaussianProcess* describing an first-order Gauss-Markov
  process. The autocovariance function is
    
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
  return gauss.gpexp((0.0,coeff,cls))


def gpbm(w):
  ''' 
  Returns brownian motion with scale parameter *w*
  '''
  def mean(x):
    out = np.zeros(x.shape[0])  
    return out 

  def cov(x1,x2):
    out = np.min(np.meshgrid(x1[:,0],x2[:,0],indexing='ij'),axis=0)
    return w**2*out
  
  return gauss.GaussianProcess(mean,cov,dim=1)  


def gpibm(w):
  ''' 
  Returns integrated brownian motion with scale parameter *w*
  '''
  def mean(x,diff):
    '''mean function which is zero for all derivatives'''
    out = np.zeros(x.shape[0])
    return out
  
  def cov(x1,x2,diff1,diff2):
    '''covariance function and its derivatives'''
    x1,x2 = np.meshgrid(x1[:,0],x2[:,0],indexing='ij')
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

  
  
