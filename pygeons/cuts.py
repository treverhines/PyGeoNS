#!/usr/bin/env python
import numpy as np

class TimeCut:
  def __init__(self,time,center=None,radius=None):
    ''' 
    Parameters
    ----------
      time: time of the discontinuity
      center: spatial center of the discontinuity
      radius: spatial radius of the discontinuity
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

class SpaceCut:
  def __init__(self,end_point1,end_point2,start=None,stop=None):
    ''' 
    Parameters
    ----------
      end_point1: first end point of the spatial discontinuity
      end_point2: second end point of the spatial discontinuity
      start: start time of the spatial discontinuity
      stop: end time of the spatial discontinuity
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


class TimeCutCollection:
  ''' 
  container for TimeCut instances
  '''
  def __init__(self,cuts=None):  
    if cuts is None:
      cuts = []

    self.cuts = cuts

  def add(self,cut):
    self.cuts += [cut]

  def get_vert_smp(self,x):
    ''' 
    returns the vertices and simplices of all cuts that x is 
    sufficiently close to
    '''
    vert = np.zeros((0,1),dtype=float)
    smp = np.zeros((0,1),dtype=int)
    for i,c in enumerate(self.cuts):
      verti,smpi = c.get_vert_smp(x)
      vert = np.vstack((vert,verti))
      smpi += smp.size
      smp = np.vstack((smp,smpi)) 

    return vert,smp      

class SpaceCutCollection:
  ''' 
  container for SpaceCut instances
  '''
  def __init__(self,cuts=None):  
    if cuts is None:
      cuts = []

    self.cuts = cuts

  def add(self,cut):
    self.cuts += [cut]

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


def load_space_cut_file(file_name,bm):
  ''' 
  space cut file contains 6 columns:

    lon1 lat1 lon2 lat2 start stop

  lonX and latX are the coordinates of the discontinuity end points

  times outside of the bounds, start and stop, will not have the 
  spatial discontinuity
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  cc = SpaceCutCollection()
  for d in data:
    x1,y1 = bm(d[0],d[1]) 
    x2,y2 = bm(d[2],d[3]) 
    c = SpaceCut([x1,y1],[x2,y2],d[4],d[5])
    cc.add(c)

  return cc

def load_time_cut_file(file_name,bm):
  ''' 
  time cut file contains 4 columns:

    time lon lat radius 

  time is the time of the discontinuity 

  stations outside the circle specified by lon,lat and radius will not 
  have the time discontinuity 
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  cc = TimeCutCollection()
  for d in data:
    x,y = bm(d[1],d[2]) 
    c = TimeCut(d[0],[x,y],d[3])
    cc.add(c)

  return cc


  return

