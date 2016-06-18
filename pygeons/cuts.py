#!/usr/bin/env python
from __future__ import division
import numpy as np

class _TimeCut:
  def __init__(self,time,center=None,radius=None):
    ''' 
    Parameters
    ----------
      time: float
        time of the discontinuity

      center: (2,) array, optional
        spatial center of the discontinuity

      radius: float, optional
        spatial radius of the discontinuity
    '''
    self.time = time
    if center is None:
      center = [0.0,0.0]

    if radius is None:
      radius = np.inf

    self.center = center 
    self.radius = radius
    return

  def exists(self,pos):
    ''' 
    returns True if the time discontinuity exists at this location
    '''
    r = np.sqrt((pos[0] - self.center[0])**2 +
                (pos[1] - self.center[1])**2)
    if r < self.radius:
      return True
    else:
      return False

  def get_vert_smp(self,pos):
    ''' 
    returns the vertices and simplices of the discontinuity if pos is 
    sufficiently close
    '''
    if self.exists(pos):
      vert = np.array([[self.time]])
      smp = np.array([[0]])
    else:
      vert = np.zeros((0,1),dtype=float)
      smp = np.zeros((0,1),dtype=int)

    return vert,smp

class _SpaceCut:
  def __init__(self,end_point1,end_point2,start=None,stop=None):
    ''' 
    Parameters
    ----------
      end_point1: (2,) array
        first end point of the spatial discontinuity

      end_point2: (2,) array
        second end point of the spatial discontinuity

      start: float
        start time of the spatial discontinuity

      stop: float
        end time of the spatial discontinuity
    '''
    self.end_point1 = end_point1
    self.end_point2 = end_point2
    if start is None:
      start = -np.inf
    if stop is None:
      stop = np.inf

    self.start = start
    self.stop = stop
    return

  def exists(self,time):
    ''' 
    returns True if the spatial discontinuity exists during this time 
    '''
    if (time >= self.start) & (time < self.stop):
      return True
    else:
      return False

  def get_vert_smp(self,time):
    ''' 
    returns the vertices and simplices of the discontinuity if time is 
    sufficiently close
    '''
    if self.exists(time):
      vert = np.array([self.end_point1,self.end_point2])
      smp = np.array([[0,1]])

    else:
      vert = np.zeros((0,2),dtype=float)
      smp = np.zeros((0,2),dtype=int)

    return vert,smp 

class TimeCuts:
  ''' 
  used to indicate cuts along time dimension
  '''
  def __init__(self,times=None,centers=None,radii=None):  
    ''' 
    Parameters
    ----------
      time : (N,) array
      
      centers : (N,2) array, optional
      
      radii : (N,) array, optional
    '''  
    if times is None:
      times = []

    if centers is None:
      centers = [None for t in times]  

    if radii is None:
      radii = [None for t in times]  

    cuts = [_TimeCut(t,c,r) for t,c,r in zip(times,centers,radii)]
    self.cuts = cuts

  def get_vert_smp(self,x):
    ''' 
    returns the vertices and simplices of all cuts that x is 
    sufficiently close to
    
    Parameters
    ----------
      x : (2,) array
      
    Returns
    -------
      vert : (K,1) float array    

      smp : (K,1) int array    

    '''
    vert = np.zeros((0,1),dtype=float)
    smp = np.zeros((0,1),dtype=int)
    for i,c in enumerate(self.cuts):
      verti,smpi = c.get_vert_smp(x)
      vert = np.vstack((vert,verti))
      smpi += smp.size
      smp = np.vstack((smp,smpi)) 

    return vert,smp      

  def __str__(self):
    out = '<TimeCuts instance with %s entries>' % len(self.cuts) 
    return out

  def __repr__(self):    
    return self.__str__()  

class SpaceCuts:
  ''' 
  used to indicate cuts in the spatial dimensions 
  '''
  def __init__(self,end_points1=None,end_points2=None,
               starts=None,stops=None):  
    ''' 
    Parameters
    ----------
      end_points1 : (N,2) array

      end_points2 : (N,2) array
      
      start : (N,) array, optional
      
      stop : (N,) array, optional
      
    ''' 
    if end_points1 is None:
      end_points1 = []

    if end_points2 is None:
      end_points2 = []
      
    if starts is None:
      starts = [None for e in end_points1] 

    if stops is None:
      stops = [None for e in end_points1] 

    cuts = [_SpaceCut(e1,e2,str,stp) for e1,e2,str,stp in 
            zip(end_points1,end_points2,starts,stops)]
    self.cuts = cuts

  def get_vert_smp(self,x):
    ''' 
    returns the vertices and simplices of all cuts that x is 
    sufficiently close to
    '''
    vert = np.zeros((0,2),dtype=float)
    smp = np.zeros((0,2),dtype=int)
    for i,c in enumerate(self.cuts):
      verti,smpi = c.get_vert_smp(x)
      vert = np.vstack((vert,verti))
      smpi += smp.size
      smp = np.vstack((smp,smpi)) 

    return vert,smp      

  def __str__(self):
    out = '<SpaceCuts instance with %s entries>' % len(self.cuts) 
    return out

  def __repr__(self):    
    return self.__str__()  
    

def load_space_cut_file(file_name,bm):
  ''' 
  space cut file contains 6 columns:
    lon1 lat1 lon2 lat2 start stop

  lonX and latX are the coordinates of the discontinuity end points

  times outside of the bounds, start and stop, will not have the 
  spatial discontinuity
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  # TODO
  return

def load_time_cut_file(file_name,bm):
  ''' 
  time cut file contains 4 columns:

    time lon lat radius 

  time is the time of the discontinuity 

  stations outside the circle specified by lon,lat and radius will not 
  have the time discontinuity 
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  # TODO
  return 

