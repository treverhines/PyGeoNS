#!/usr/bin/env python
# In this script we solve the least squares problem
#
#   |   W |     | W*d |
#   | p*L | u = |   0 |
#
# where W is a weight matrix composed of the uncertainties in the 
# observations d, L is a second order differential operator, and p is 
# a penalty parameter. We explore the effect of p on the covariance of u.
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import rbf.fd
import scipy.sparse
import scipy.signal
import matplotlib.cm

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

# number of observations
Nt = 10001
# number of trials
Np = 1000
sigma = 1.0
order = 2
omega = 0.1
p = 1.0/(sigma*(2*np.pi*omega)**order)

t = np.linspace(-100.0,100.0,Nt)
dt = t[1] - t[0]

d = np.random.normal(0.0,sigma,(Nt,Np)) 

L = rbf.fd.poly_diff_matrix(t[:,None],diff=(order,))
lhs = scipy.sparse.eye(Nt)/sigma**2 + p**2*L.T.dot(L)
rhs = d/sigma**2
u = scipy.sparse.linalg.spsolve(lhs,rhs)

fig,ax = plt.subplots()
plt.plot(t,u)

fig,ax = plt.subplots()
freq,pow = psd(u.T,t)
filter = (1.0/sigma**2)/((1.0/sigma**2) + p**2*(2*np.pi*freq)**(2*order))
ax.loglog(freq,pow,'k')
ax.loglog(freq,filter**2)
''' 
quit()
freq = np.fft.fftfreq(Nt,dt)
filter = (1.0/sigma**2)/((1.0/sigma**2) + p**2*(2*np.pi*freq)**(2*order))
dhat = np.array([np.fft.fft(i) for i in d.T])
uhat = np.array([filter*i for i in dhat])
u = np.array([np.fft.ifft(i) for i in uhat])

fig,ax = plt.subplots()
plt.plot(t,u.T)

fig,ax = plt.subplots()
freq,pow = psd(u,t)
ax.loglog(freq,pow,'k')
ax.loglog(freq,filter**2)
'''
plt.show()
quit()


autocov = np.fft.ifft(np.fft.fftshift(pow))
plt.plot(t,np.fft.fftshift(autocov))

quit()
# penalty parameters
p = 1.0

u = scipy.sparse.linalg.spsolve(lhs,rhs)

# compute correlation matrix
C = np.corrcoef(u)
fig,ax = plt.subplots()
ax.plot(times,C[Nt//2,:],lw=2)
#ax.plot(times,np.sinc((times)/(np.pi*T)))
ax.grid()

#for i in [0,Nt//4,Nt//2,3*Nt//4,Nt-1]:
#  ax.plot(times,C[i,:],color=matplotlib.cm.spectral(i/Nt),lw=2)
freq = np.fft.fftfreq(Nt,dt)
coeff = np.fft.fft(C[Nt//2,:])
pow = coeff.real
#freq,pow = scipy.signal.periodogram(C[Nt//2,:],1.0/dt,
#                                    return_onesided=False,scaling='spectrum')
np.fft.fft(C[Nt//2,:])

pow_true = (1.0/(1 + (2*np.pi*freq)**(2*order)))
#freq,pow = scipy.signal.periodogram(np.sinc(times/(np.pi*T)),1.0/dt)
fig,ax = plt.subplots()
ax.plot(freq,pow)
ax.plot(freq,pow_true)
ax.grid()

#plt.plot(times,C[Nt//2,:],'k')
#plt.plot(times,C[Nt//4,:],'k')
plt.show()


