#!/usr/bin/env python
import numpy as np

class TimeCut:
  def __init__(self,time,pos_center,pos_radius):
    ''' 
    Parameters
    ----------
      time: (scalar) time of the discontinuity
      pos_center: (2-array) spatial center of the discontinuity
      pos_radius: (scalar) spatial radius of the discontinuity
    '''
    self.time = time
    self.pos_center = pos_center 
    self.pos_radius = pos_radius
    return

  def is_cut(self,pos):
    ''' 
    tests whether pos is sufficiently close to the discontinuity
    '''
    r = np.sqrt((pos[0] - self.pos_center[0])**2 +
                (pos[1] - self.pos_center[1])**2)
    if r < self.pos_radius:
      return True
    else:
      return False

  def get_vert_smp(self,pos):
    ''' 
    returns the vertices and simplices of the discontinuity if pos is 
    sufficiently close
    '''
    if self.is_cut(pos):
      vert = np.array([[self.time]])
      smp = np.array([[0]])
    else:
      vert = np.zeros((0,1),dtype=float)
      smp = np.zeros((0,1),dtype=int)

    return vert,smp

class SpaceCut:
  def __init__(self,pos1,pos2,time_center,time_radius):
    ''' 
    Parameters
    ----------
      pos1: (2-array) first vertex of the spatial discontinuity
      pos2: (2-array) second vertex of the spatial discontinuity
      time_center: (scalar) time center of the discontinuity
      time_radius: (scalar) time radius of the discontinuity
    '''
    self.pos1 = pos1
    self.pos2 = pos2
    self.time_center = time_center
    self.time_radius = time_radius
    return

  def is_cut(self,time):
    ''' 
    returns True if the time is sufficiently close to the discontinuity
    '''
    r = abs(time - self.time_center)
    if r < self.time_radius:
      return True
    else:
      return False

  def get_vert_smp(self,time):
    ''' 
    returns the vertices and simplices of the discontinuity if time is 
    sufficiently close
    '''
    if self.is_cut(time):
      vert = np.array([self.pos1,self.pos2])
      smp = np.array([[0,1]])

    else:
      vert = np.zeros((0,2),dtype=float)
      smp = np.zeros((0,2),dtype=int)

    return vert,smp 


class CutCollection:
  ''' 
  container for either TimeCuts or SpaceCuts
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
    vert = None
    smp = None
    for i,c in enumerate(self.cuts):
      verti,smpi = c.get_vert_smp(x)
      if vert is None:
        vert = verti 
      else:
        vert = np.vstack((vert,verti))

      if smp is None:
        smp = smpi
      else:
        smpi += smp.size
        smp = np.vstack((smp,smpi)) 

    return vert,smp      


def load_space_cut_file(file_name,bm):
  ''' 
  space cut file contains 6 columns:

    pos1_lon pos1_lat pos2_lon pos2_lat time_center time_radius 

  posX_lon and posX_lat are the coordinates of the discontinuity vertices

  time_center and time_radius specify the temporal duration of the 
  spatial discontinuity
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  cc = CutCollection()
  for d in data:
    x1,y1 = bm(d[0],d[1]) 
    x2,y2 = bm(d[2],d[3]) 
    c = SpaceCut([x1,y1],[x2,y2],d[4],d[5])
    cc.add(c)

  return cc

def load_time_cut_file(file_name,bm):
  ''' 
  time cut file contains 6 columns:

  time pos_lon pos_lat pos_radius 
  ''' 
  data = np.loadtxt(file_name,skiprows=1,dtype=float)
  cc = CutCollection()
  for d in data:
    x,y = bm(d[1],d[2]) 
    c = TimeCut(d[0],[x,y],d[3])
    cc.add(c)

  return cc


  return

