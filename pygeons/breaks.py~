''' 
This module provides functions for converting the descriptions of 
temporal and spatial discontinuities. The functions convert the user 
input of discontinuities into a simplicial complex defined by vertices 
and simplices. The latter is used for specifying discontinuities in 
the RBF package.
'''
import numpy as np
from pygeons.mjd import mjd

def make_time_vert_smp(break_dates,fmt='%Y-%m-%d'):
  ''' 
  Returns the vertices and simplices defining the time breaks
 
  Parameters
  ----------
  break_dates : (N,) str array
    List of string representation of the dates where there are time 
    discontinuities. The date should be for the first day that the 
    discontinuity is observed.

  Returns
  -------
  vert : (N,1) float array
    Times, in MJD, of the discontinuities
    
  smp : (N,1) int array
    Array of indices 1 through N.  
  '''
  if break_dates is None: break_dates = []
  # subtract half a day to get rid of any ambiguity about what day 
  # the dislocation is observed
  breaks = [mjd(d,fmt) - 0.5 for d in break_dates]
  vert = np.array(breaks).reshape((-1,1))
  smp = np.arange(vert.shape[0]).reshape((-1,1))
  return vert,smp
  

def make_space_vert_smp(break_lons,break_lats,break_conn,bm):
  '''  
  Returns the vertices and simplices defining the space breaks
  
  Parameters
  ----------
  break_lons : (N,) float array
    List of longitudes for each vertex in the spatial discontinuities
    
  break_lats : (N,) float array
    List of latitudes for each vertex in the spatial discontinuities
      
  break_conn : (M,) str array 
    List of strings indicating the how the vertices are connected to 
    form each discontinuity. For example, ['0-1-2','3-4'], indicates 
    that vertices 0, 1, and 2 form one spatial discontinuity and 
    vertices 3 and 4 form another.

  bm : Basemap instance
    Maps longitude and latitude to meters north and meters east

  Returns
  -------
  vert : (N,2) float array
    Array of vertices making up the discontinuities in meters north 
    and meters east.
    
  smp : (Q,2) int array
    Array of indices indicating which vertices are connected

  '''
  if break_lons is None:
    break_lons = np.zeros(0,dtype=float)
  else:
    break_lons = np.asarray(break_lons,dtype=float)

  if break_lats is None:
    break_lats = np.zeros(0,dtype=float)
  else:
    break_lats = np.asarray(break_lats,dtype=float)

  if break_lons.shape[0] != break_lats.shape[0]:
    raise ValueError('*break_lons* and *break_lats* must have the same length')

  N = break_lons.shape[0]
  if break_conn is None:
    if N != 0:
      break_conn = ['-'.join(np.arange(N).astype(str))]
    else:
      break_conn = []

  smp = []
  for c in break_conn:
    idx = np.array(c.split('-'),dtype=int)
    smp += zip(idx[:-1],idx[1:])

  smp = np.array(smp,dtype=int).reshape((-1,2))
  vert = [bm(i,j) for i,j in zip(break_lons,break_lats)]
  vert = np.array(vert,dtype=float).reshape((-1,2))
  return vert,smp
  

