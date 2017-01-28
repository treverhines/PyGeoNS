#!/usr/bin/env python
import numpy as np
import pygeons.diff
import pygeons.smooth
import pygeons.cuts
import pygeons.quiver
import matplotlib.pyplot as plt
import logging
import rbf.halton
import rbf.basis
import pygeons.view
rbf.basis.set_sym_to_num('numpy')
import scipy.sparse
logging.basicConfig(level=logging.DEBUG)

def psd(signals,times):
  ''' 
  
  returns the power spectral density using N realizations of a signal
  
  Parameters
  ----------
  signals : (N,Nt) array
  
  times : (Nt,) array

  Returns
  -------
    freq : (Nt,) array

    pow : (Nt,) array
  '''
  signals = np.asarray(signals)
  times = np.asarray(times)
  Nt = times.shape[0]
  dt = times[1] - times[0]
  freq = np.fft.fftfreq(Nt,dt)
  # normalize the coefficients by 1/sqrt(Nt)
  coeff = np.array([np.fft.fft(i)/np.sqrt(Nt) for i in signals])
  # get the complex modulus
  pow = coeff*coeff.conj()
  # get the expected value
  pow = np.mean(pow,axis=0)
  return freq,pow

  
cutoff = 1.0/10.0
Nt = 50000
t = np.linspace(0.0,5000.0,Nt)
dt = t[1] - t[0]
x = np.array([[0.0,0.0]])
S = 1.0
sigma = S*np.ones((t.shape[0],x.shape[0]))
u = np.random.normal(0.0,S,(500,Nt,1))

ds = pygeons.diff.acc()
ds['time']['diffs'] = [[2]]
us = pygeons.smooth.smooth(t,x,u,ds,sigma=sigma,time_cutoff=cutoff)

fig,ax = plt.subplots()
ax.plot(t,us[0,:,0],'b-')
#ax.plot(t,u[:,0],'ko')

# plot spectral density
fig,ax = plt.subplots()
freq,pow = psd(us[:,:,0],t)
ax.loglog(freq,pow)
filter = 1.0/(1.0 + (freq/cutoff)**(2*2))
ax.loglog(freq,filter**2)

plt.show()
