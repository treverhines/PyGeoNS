#!/usr/bin/env python
import numpy as np
import pygeons.cuts
import pygeons.smooth
import rbf.halton
import matplotlib.pyplot as plt
import modest

Nt = 5
Nx = 20
t = np.linspace(0.0,1.0,Nt)
pos = rbf.halton.halton(Nx,2)

sc = pygeons.cuts.SpaceCut([0.501,0.0],[0.501,1.0],0.5,10.1)
scc = pygeons.cuts.SpaceCutCollection([sc])

tc = pygeons.cuts.TimeCut(0.5,[0.0,0.0],0.5)
tcc = pygeons.cuts.TimeCutCollection([tc])
tcc = None

modest.tic()
Lt,Lx = pygeons.smooth._reg_matrices(t,pos,space_cuts=scc,time_cuts=tcc)
print(modest.toc())
modest.tic()
G = pygeons.smooth._system_matrix(Nt,Nx)
print(modest.toc())
plt.figure(1)
plt.imshow(Lt.toarray(),interpolation='none')
plt.figure(2)
plt.imshow(Lx.toarray(),interpolation='none')
plt.figure(3)
plt.imshow(G.toarray(),interpolation='none')
plt.show()
