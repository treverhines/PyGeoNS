#!/usr/bin/env python
from __future__ import division
import numpy as np
from pygeons.view import InteractiveViewer
from pygeons.downsample import weighted_mean
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
import logging
logger = logging.getLogger(__name__)

def most(a,axis=None,cutoff=0.5):
  ''' 
  behaves like np.all but returns True if the percentage of Trues
  is greater than or equal to cutoff
  '''
  a = np.asarray(a,dtype=bool)
  if axis is None:
    b = np.prod(a.shape)
  else:
    b = a.shape[axis]

  asum = np.sum(a,axis=axis)
  return asum >= b*cutoff


def time_lacks_data(sigma,cutoff=0.5):
  ''' 
  returns true for each time where sigma is np.inf for most of the stations.
  The percentage of infs must exceed cutoff in order to return True.
  '''
  out = most(np.isinf(sigma),axis=1,cutoff=cutoff)
  return out


def station_lacks_data(sigma,cutoff=0.5):
  ''' 
  returns true for each station where sigma is np.inf for most of the times.
  The percentage of infs must exceed cutoff in order to return True.
  '''
  out = most(np.isinf(sigma),axis=0,cutoff=cutoff)
  return out


def station_is_duplicate(sigma,x,tol=4.0):
  ''' 
  returns the indices for a set of nonduplicate stations. if duplicate 
  stations are found then the one that has the most observations is 
  retained
  '''
  x = np.asarray(x)
  # if there is zero or one station then dont run this check

  is_duplicate = np.zeros(x.shape[0],dtype=bool)
  while True:
    xi = x[~is_duplicate]
    sigmai = sigma[:,~is_duplicate]
    idx = _identify_duplicate_station(sigmai,xi,tol=tol) 
    if idx is None:
      break
    else:  
      global_idx = np.nonzero(~is_duplicate)[0][idx]
      logger.info('identified station %s as a duplicate' % global_idx)
      is_duplicate[global_idx] = True

  return is_duplicate      


def _identify_duplicate_station(sigma,x,tol=3.0):
  ''' 
  returns the index of the station which is likely to be a duplicate 
  due to its proximity to another station.  The station which has the 
  least amount of data is identified as the duplicate
  '''
  # if there is zero or one station then dont run this check
  if x.shape[0] <= 1:
    return None
    
  T = cKDTree(x)
  dist,idx = T.query(x,2)
  r = dist[:,1]
  ri = idx[:,1]
  logr = np.log10(r)
  cutoff = np.mean(logr) - tol*np.std(logr)
  if not np.any(logr < cutoff):
    # if no duplicates were found then return nothing
    return None

  else:
    # otherwise return the index of the least useful of the two 
    # nearest stations
    idx1 = np.argmin(logr)    
    idx2 = ri[idx1]
    count1 = np.sum(~np.isinf(sigma[:,idx1]))
    count2 = np.sum(~np.isinf(sigma[:,idx2]))
    count,out = min((count1,idx1),(count2,idx2))
    return out
  

class InteractiveCleaner(InteractiveViewer):
  ''' 
               ---------------------------------
               PyGeoNS Interactive Cleaner (PIC)
               ---------------------------------

Controls
--------
    Enter : edit the configurable parameters through the command line. 
        Variables can be defined using any functions in the numpy, 
        matplotlib, or base python namespace

    Left : move back 1 time step (Ctrl-Left and Alt-Left move back 10 
        and 100 respectively)

    Right : move forward 1 time step (Ctrl-Right and Alt-Right move 
        forward 10 and 100 respectively)

    Up : move forward 1 station (Ctrl-Left and Alt-Left move back 10 
        and 100 respectively)
          
    Down : move back 1 station (Ctrl-Right and Alt-Right move forward 
        10 and 100 respectively)
          
    R : redraw figures

    H : hide station marker

    D : enables outlier removal mode while pressed.  Click and drag on 
        the time series axes to remove outliers within a time interval

    J : enables jump removal mode while pressed. Jumps are estimated 
        by taking a weighted mean of the data over a time interval 
        before and after the jump. Click on the time series axes to 
        identify a jump and drag the cursor over the desired time 
        interval.
        
    K : keep edited data
    
    U : undo edits since data was last kept
    
    W : write kept changes to an hdf5 file
    
Notes
-----
    Stations may also be selected by clicking on them 

    Exit PIC by closing the figures

    Key bindings only work when the active window is one of the PIC
    figures   

---------------------------------------------------------------------     
  '''
  def __init__(self,t,x,
               u=None,v=None,z=None,
               su=None,sv=None,sz=None,
               **kwargs):
    ''' 
    Parameters
    ----------
      t : (Nt,) array
        observation times
        
      x : (Nx,2) array
        observation positions
        
      u : (...,Nt,Nx) array
        east component

      v : (...,Nt,Nx) array
        north component

      z : (...,Nt,Nx) array
        vertical component

      su : (Nt,Nx) array, optional
        standard deviation of east component

      sv : (Nt,Nx) array, optional
        standard deviation of north component

      sz : (Nt,Nx) array, optional
        standard deviation of vertical component
        
    Note
    ----
      only one of u, v, and z need to be specified
    '''
    Nt,Nx = len(t),len(x)
    if u is not None:
      u = np.asarray(u)
      input_shape = u.shape
      u_view = [u.reshape((-1,Nt,Nx))[0]]
    else:
      u_view = None
      
    if v is not None:
      v = np.asarray(v)
      input_shape = v.shape
      v_view = [v.reshape((-1,Nt,Nx))[0]]
    else:
      v_view = None
      
    if z is not None:
      z = np.asarray(z)
      input_shape = z.shape
      z_view = [z.reshape((-1,Nt,Nx))[0]]
    else:
      z_view = None
      
    if su is not None:
      su = [np.asarray(su)]

    if sv is not None:
      sv = [np.asarray(sv)]

    if sz is not None:
      sz = [np.asarray(sz)]
      
    data_set_names = kwargs.pop('data_set_names',['edited data'])
    color_cycle = kwargs.pop('color_cycle',['k'])
    InteractiveViewer.__init__(self,t,x,
                               u=u_view,v=v_view,z=z_view,
                               su=su,sv=sv,sz=sz,
                               color_cycle=color_cycle,
                               data_set_names=data_set_names,
                               **kwargs)
                               
    if u is None:
      u = np.zeros(input_shape)
      
    if v is None:
      v = np.zeros(input_shape)

    if z is None:
      z = np.zeros(input_shape)
      
    self.input_shape = input_shape

    # all changes made to the viewed data set should be broadcasted onto 
    # this one. 
    tpl = (u[...,None],v[...,None],z[...,None])
    all_data = np.concatenate(tpl,axis=-1)
    # flatten the broadcast dimensions
    all_data = all_data.reshape((-1,Nt,Nx,3))

    # set any masked data to nan
    all_data[:,np.isinf(self.sigma_sets[0])] = np.nan
     
    self.all_data_sets = [all_data]
    self._mode = None
    self._is_pressed = False

  def get_data(self):
    all_data = self.all_data_sets[0]
    # set any masked data to nan. This should already be the case and 
    # I am probably being too cautious
    all_data[:,np.isinf(self.sigma_sets[0])] = np.nan
    
    all_data = all_data.reshape(self.input_shape+(3,))
    u = all_data[...,0]
    v = all_data[...,1]
    z = all_data[...,2]
    su = self.sigma_sets[0][:,:,0]
    sv = self.sigma_sets[0][:,:,1]
    sz = self.sigma_sets[0][:,:,2]
    return (u,v,z,su,sv,sz)
    
  def connect(self):
    self.ts_fig.canvas.mpl_connect('button_press_event',self.on_mouse_press)
    self.ts_fig.canvas.mpl_connect('motion_notify_event',self.on_mouse_move)
    self.ts_fig.canvas.mpl_connect('button_release_event',self.on_mouse_release)
    self.ts_fig.canvas.mpl_connect('key_release_event',self.on_key_release)
    self.map_fig.canvas.mpl_connect('key_release_event',self.on_key_release)
    InteractiveViewer.connect(self)
        
  def remove_jump(self,jump_time,radius):
    xidx = self.config['xidx']
    tidx_right, = np.nonzero((self.t > jump_time) & 
                             (self.t <= (jump_time+radius)))
    tidx_left, = np.nonzero((self.t < jump_time) & 
                            (self.t >= (jump_time-radius)))
    if (len(tidx_right) == 0) | (len(tidx_left) == 0):
      name = self.station_labels[xidx]
      logger.info('cannot remove jump for station %s due to a lack of data\n' % name)
      return

    data = self.all_data_sets[0]
    # expand sigma to the size of data
    sigma = self.sigma_sets[0][None,:,:].repeat(data.shape[0],axis=0)

    mean_right,sigma_right = weighted_mean(data[:,tidx_right,xidx],
                                           sigma[:,tidx_right,xidx],
                                           axis=1)
    mean_left,sigma_left = weighted_mean(data[:,tidx_left,xidx],
                                         sigma[:,tidx_left,xidx],
                                         axis=1)
    # jump for each component
    jump = mean_right - mean_left
    
    # uncertainty in the jump estimate
    jump_sigma = np.sqrt(sigma_right**2 + sigma_left**2)

    # find indices of all times after the jump
    all_tidx_right, = np.nonzero(self.t > jump_time)
    
    # remove jump from values make after the jump 
    new_pos = data[:,all_tidx_right,xidx,:] - jump[:,None,:]
    
    # increase uncertainty 
    new_var = sigma[:,all_tidx_right,xidx,:]**2 + jump_sigma[:,None,:]**2
    new_sigma = np.sqrt(new_var)
    
    self.all_data_sets[0][:,all_tidx_right,xidx,:] = new_pos
    self.data_sets[0][all_tidx_right,xidx,:] = new_pos[0]
    self.sigma_sets[0][all_tidx_right,xidx,:] = new_sigma[0]
    
    # turn the new data sets into masked arrays
    self.update_data()

    name = self.station_labels[xidx]
    logger.info('removed jump at time %g for station %s using data from time %g to %g\n' % 
                (jump_time,name,jump_time-radius,jump_time+radius))
      
  def remove_outliers(self,start_time,end_time):
    # this function masks data for the current station which ranges 
    # from start_time to end_time
    xidx = self.config['xidx']
    tidx, = np.nonzero((self.t >= start_time) & (self.t <= end_time))
    self.all_data_sets[0][:,tidx,xidx] = np.nan
    self.data_sets[0][tidx,xidx] = np.nan
    self.sigma_sets[0][tidx,xidx] = np.inf

    # turn the new data sets into masked arrays
    self.update_data()

    name = self.station_labels[xidx]
    logger.info('removed data from time %g to %g for station %s\n' % (start_time,end_time,name))
          
  def on_mouse_press(self,event):
    # ignore if not the left button
    if event.button != 1: return
    # ignore if the event was not in an axis
    if event.inaxes is None: return
    # ignore if the event was not in the time series figure
    if not event.inaxes.figure is self.ts_fig: return

    self._is_pressed = True
    self._t1 = event.xdata
    self.rects = []
    self.vlines = []
    for ax in self.ts_ax:
      ymin,ymax = ax.get_ylim()
      r = Rectangle((self._t1,ymin),0.0,ymax-ymin,color='none',alpha=0.5,edgecolor=None)
      self.rects += [r]
      self.vlines += [ax.vlines(self._t1,ymin,ymax,color='none')]
      ax.add_patch(r)
          
    self.ts_fig.canvas.draw()  
        
  def on_mouse_move(self,event):
    # do nothing is a mouse button is not being held down
    if not self._is_pressed: return
    # ignore if the event was not in an axis
    if event.inaxes is None: return
    # ignore if the event was not in the time series figure
    if not event.inaxes.figure is self.ts_fig: return

    self._t2 = event.xdata
    for r,v in zip(self.rects,self.vlines):
      if self._mode == 'OUTLIER_REMOVAL':
        r.set_width(self._t2 - self._t1) 
        r.set_x(self._t1)
        r.set_color('r')
        v.set_color('k')
        
      elif self._mode == 'JUMP_REMOVAL':
        r.set_width(2*(self._t2 - self._t1))
        r.set_x(self._t1 - (self._t2 - self._t1))
        r.set_color('b')
        v.set_color('k')
       
      else:
        r.set_width(0.0)
        r.set_color('none')
        v.set_color('none')
            
    self.ts_fig.canvas.draw()  

  def on_mouse_release(self,event):
    # ignore if not the left button
    if event.button != 1: return
    # do nothing is a mouse button was not clicked in the axis
    if not self._is_pressed: return

    self._is_pressed = False
    # remove the rectangles no matter where the button was released
    for r,v in zip(self.rects,self.vlines):
      r.remove()
      v.remove()

    self.ts_fig.canvas.draw()  

    # only act on this event if the following conditions are met
    # ignore if the event was not in an axis
    if event.inaxes is None: return
    # ignore if the event was not in the time series figure
    if not event.inaxes.figure is self.ts_fig: return

    self._t2 = event.xdata
    # act according to the self._mode at the time of release
    if self._mode == 'OUTLIER_REMOVAL':
      mint = min(self._t1,self._t2)
      maxt = max(self._t1,self._t2)
      self.remove_outliers(mint,maxt) 
      # keep the time series axis limits fixed
      xlims = [i.get_xlim() for i in self.ts_ax]      
      ylims = [i.get_ylim() for i in self.ts_ax]      
      self.update()   
      [i.set_xlim(j) for i,j in zip(self.ts_ax,xlims)]
      [i.set_ylim(j) for i,j in zip(self.ts_ax,ylims)]
      self.ts_fig.canvas.draw()  

    elif self._mode == 'JUMP_REMOVAL':
      self.remove_jump(self._t1,abs(self._t2-self._t1))
      # keep the time series axis limits fixed
      xlims = [i.get_xlim() for i in self.ts_ax]      
      ylims = [i.get_ylim() for i in self.ts_ax]      
      self.update()   
      [i.set_xlim(j) for i,j in zip(self.ts_ax,xlims)]
      [i.set_ylim(j) for i,j in zip(self.ts_ax,ylims)]
      self.ts_fig.canvas.draw()  
    
    else: 
      return

  def _set_mode(self,name):
    if self._mode is None:
      self._mode = name
      print('enabled %s mode\n' % name) 

  def _unset_mode(self,name):
    if self._mode is name:
      self._mode = None
      print('disabled %s mode\n' % name) 
  
  def on_key_press(self,event):
    print('pressed %s' % event.key)
    # disable c
    if event.key == 'c':
      return
              
    elif event.key == 'd':
      self._set_mode('OUTLIER_REMOVAL')
      self.on_mouse_move(event)

    elif event.key == 'j':
      self._set_mode('JUMP_REMOVAL')
      self.on_mouse_move(event)

    else:
      InteractiveViewer.on_key_press(self,event)
        
  def on_key_release(self,event):
    print('released %s' % event.key)
    if event.key == 'd':
      self._unset_mode('OUTLIER_REMOVAL')
      self.on_mouse_move(event)

    elif event.key == 'j':
      self._unset_mode('JUMP_REMOVAL')
      self.on_mouse_move(event)
  
def clean(*args,**kwargs):
  ''' 
  Runs InteractiveCleaner and returns the kept data
  '''
  ic = InteractiveCleaner(*args,**kwargs)
  ic.connect()
  plt.show()
  return ic.get_data()
  
                                      
