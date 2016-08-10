#!/usr/bin/env python
import numpy as np
import rbf.fd
import matplotlib.pyplot as plt
import scipy.sparse

def psd2d(signals,x,y):
  ''' 
  Parameters
  ---------- 
    signal : (P,Nx,Ny)  
    x : (Nx,)
    y : (Ny,)

  '''  
  signals = np.asarray(signals)
  x = np.asarray(x)
  y = np.asarray(y)
  Nx = x.shape[0]
  Ny = y.shape[0]
  dx = x[1]-x[0]
  dy = y[1]-y[0]
  freqx = np.fft.fftfreq(Nx,dx)
  freqy = np.fft.fftfreq(Ny,dy)
  coeff = np.fft.fft2(signals)/(np.sqrt(Nx)*np.sqrt(Ny))
  pow = coeff*coeff.conj()
  pow = np.mean(pow,axis=0)
  return pow,freqx,freqy
  
 
Nx = 100
Ny = 100
x = np.linspace(-5.0,5.0,Nx)
y = np.linspace(-5.0,5.0,Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
#xg,yg = np.meshgrid(x,y)


order = 3
P = 1000
sigma = 1.0
penalty = 1.0/(sigma*(2*np.pi*0.5)**order)
u = np.random.normal(0.0,sigma,(P,Nx,Ny))

# form the differentiation matrix
print('building')
idx = np.arange(Nx*Ny).reshape((Nx,Ny))
D = scipy.sparse.lil_matrix((Nx*Ny,Nx*Ny))
for i in idx.T:
  D[np.ix_(i,i)] += rbf.fd.poly_diff_matrix(x[:,None],(order,))
for i in idx:
  D[np.ix_(i,i)] += rbf.fd.poly_diff_matrix(y[:,None],(order,))
  
D = D.tocsr()
D.eliminate_zeros()
print('done')
freq1 = np.fft.fftfreq(Nx,dx)
freq2 = np.fft.fftfreq(Ny,dy)
s = (1.0/sigma)**2
f1 = (2*np.pi*freq1)**(order)
f2 = (2*np.pi*freq2)**(order)
filter = s/(s + penalty**2*(f1[:,None] + f2[None,:])**2)

u_flat = u.reshape((P,Nx*Ny))

lhs = scipy.sparse.eye(Nx*Ny)/sigma**2 + penalty**2*D.T.dot(D)
rhs = u_flat/sigma**2
u_smooth = scipy.sparse.linalg.spsolve(lhs,rhs.T).T
u_smooth = u_smooth.reshape((P,Nx,Ny))

fig,ax = plt.subplots()
plt.imshow(u_smooth[0].T,extent=(-5,5,-5,5))

pow,fx,fy = psd2d(u_smooth,x,y)
fig,ax = plt.subplots()
fmin = np.min(freq1)
fmax = np.max(freq1)
p = plt.imshow(np.fft.fftshift(pow).real,vmin=0,vmax=1,extent=(fmin,fmax,fmin,fmax),cmap='viridis')
plt.colorbar(p)
fig,ax = plt.subplots()
p = plt.imshow(np.fft.fftshift(filter)**2,vmin=0,vmax=1,extent=(fmin,fmax,fmin,fmax),cmap='viridis')
plt.colorbar(p)
fig,ax = plt.subplots()
p = plt.imshow(np.fft.fftshift(pow).real-np.fft.fftshift(filter)**2,vmin=-1.0,vmax=1.0,extent=(fmin,fmax,fmin,fmax),cmap='viridis')
plt.colorbar(p)
plt.show()
quit()
# find the frequency of the smoothed solution

plt.imshow(u_smooth1[0])
plt.show()
quit()
plt.imshow(np.fft.fftshift(filter),interpolation='none')
plt.show()
quit()

dx = x[1] - x[0]
freq = np.fft.fftfreq(Nx,dx)
coeff = np.fft.fft2(u)
filter = 1.0/(1.0 + ((freq[:,None]**N + freq[None,:]**N)/(0.5**N))**2)
coeff_smooth = coeff * filter
u_smooth = np.fft.ifft2(coeff_smooth).real
#pow = coeff*coeff.conj()
#pow = pow.real
#pow = np.mean(pow,axis=0)
#pow = np.fft.fftshift(pow)

#fig,ax = plt.subplots()
#p = ax.pcolor(np.fft.fftshift(freq),np.fft.fftshift(freq),pow,cmap='viridis')
#plt.colorbar(p)

fig,ax = plt.subplots()
p = ax.imshow(filter,cmap='viridis')
plt.colorbar(p)
fig,ax = plt.subplots()
p = ax.imshow(u_smooth,cmap='viridis')
plt.colorbar(p)
plt.show()

