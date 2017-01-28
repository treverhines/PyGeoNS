#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import rbf.halton
import rbf.basis
import modest
from pygeons.view import network_viewer

Nt = 10
Nx = 50
t = 2010 + np.linspace(0.0,1.0,Nt)
x = rbf.halton.halton(Nx,2)
data = (np.cos(2*np.pi*t[:,None]) *
        np.sin(2*np.pi*x[:,0])[None,:] *
        np.cos(2*np.pi*x[:,1])[None,:])

network_viewer(t,x,z=[data])
