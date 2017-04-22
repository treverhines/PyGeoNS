from __future__ import division
import numpy as np
from pygeons.plot.ivector import InteractiveVectorViewer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
import warnings
logger = logging.getLogger(__name__)


def weighted_mean(x,sigma,axis=None):
  ''' 
  Computes the weighted mean of *x* with uncertainties *sigma*
  
  Parameters
  ----------
    x : (..., N,...) array
    
    sigma : (..., N,...) array

    axis : int, optional
    
  Notes
  -----
  If all uncertainties along the axis are np.inf then then the
  returned mean is np.nan with uncertainty is np.inf
    
  If there are 0 entries along the axis then the returned mean is
  np.nan with uncertainty np.inf
    
  zero uncertainties will raise an error
    
  '''
  x = np.array(x,copy=True)
  sigma = np.asarray(sigma)
  # convert any nans to zeros
  x[np.isnan(x)] = 0.0
  if x.shape != sigma.shape:
    raise ValueError('x and sigma must have the same shape')

  # make sure uncertainties are positive
  if np.any(sigma <= 0.0):
    raise ValueError('uncertainties must be positive and nonzero')

  numer = np.sum(x/sigma**2,axis=axis)
  denom = np.sum(1.0/sigma**2,axis=axis)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # out_value can be nan if the arrays have zero length along axis 
    # or if sigma is inf. out_sigma will be inf in that case 
    out_value = numer/denom
    out_sigma = np.sqrt(1.0/denom)

  return out_value,out_sigma



class InteractiveCleaner(InteractiveVectorViewer):
  ''' 
----------------- PyGeoNS Interactive Cleaner (PIC) -----------------

An interactive figure for removing outliers and time series jumps.

Controls :
  Left : Move back 1 time step (Ctrl-Left and Alt-Left move back 10
    and 100 respectively).
  
  Right : Move forward 1 time step (Ctrl-Right and Alt-Right move
    forward 10 and 100 respectively).
  
  Up : Move forward 1 station (Ctrl-Left and Alt-Left move back 10 and
    100 respectively).
            
  Down : Move back 1 station (Ctrl-Right and Alt-Right move forward 10
    and 100 respectively).
            
  R : Redraw figures.
  
  V : Toggle whether to hide the vertical deformation.
  
  H : Toggle whether to hide the station marker.
  
  D : Enables outlier removal mode while pressed.  Click and drag on
    the time series axes to remove outliers within a time interval.
  
  J : Enables jump removal mode while pressed. Jumps are estimated by
    taking a weighted mean of the data over a time interval before and 
    after the jump. Click on the time series axes to identify a jump and 
    drag the cursor over the desired time interval.

  Enter : Edit the configurable parameters through the command line.
    Variables can be defined using any functions in the numpy, 
    matplotlib, or base python namespace.
      
Notes
-----
  Stations may also be selected by clicking on them.

  Exit PIC by closing the figures.

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
        
      u : (Nt,Nx) array
        east component

      v : (Nt,Nx) array
        north component

      z : (Nt,Nx) array
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
    dataset_labels = kwargs.pop('dataset_labels',['edited data'])
    InteractiveVectorViewer.__init__(self,t,x,
                                     u=[u],v=[v],z=[z],
                                     su=[su],sv=[sv],sz=[sz],
                                     dataset_labels=dataset_labels,
                                     **kwargs)
    self._mode = None
    self._mouse_is_pressed = False
    # record of all removed jumps and outliers  
    self.log = []

  def get_data(self):
    u = self.data_sets[0][:,:,0]
    v = self.data_sets[0][:,:,1]
    z = self.data_sets[0][:,:,2]
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
    InteractiveVectorViewer.connect(self)
        
  def remove_jump(self,jump_time,delta):
    ''' 
    estimates and removes a jump at time *jump_time*. *jump_time* is
    an integer and indicates the first day of the jump. The jump size
    is the difference between the mean values over an interval *delta*
    before and after the jump. If no data is available over these
    intervals then no changes will be made.

    '''
    xidx = self.config['xidx']
    tidx_right, = np.nonzero((self.t >= jump_time) & 
                             (self.t <= (jump_time+delta)))
    tidx_left,  = np.nonzero((self.t <  jump_time) & 
                             (self.t >= (jump_time-delta)))
    mean_right,_ = weighted_mean(self.data_sets[0][tidx_right,xidx,:],
                                 self.sigma_sets[0][tidx_right,xidx,:],
                                 axis=0)
    mean_left,_ = weighted_mean(self.data_sets[0][tidx_left,xidx,:],
                                self.sigma_sets[0][tidx_left,xidx,:],
                                axis=0)
    # jump for each component
    jump = mean_right - mean_left
    # only remove jumps when a jump can be calculated
    finite_idx, = np.isfinite(jump).nonzero()
    all_tidx_right, = np.nonzero(self.t >= jump_time)
    new_pos = self.data_sets[0][all_tidx_right,xidx,:]
    new_pos[:,finite_idx] = new_pos[:,finite_idx] - jump[finite_idx] 
    self.data_sets[0][all_tidx_right,xidx] = new_pos
    self.log.append(('jump',xidx,jump_time,delta))
    logger.info('removed jump for station %s at time %s' % (xidx,jump_time))
      
  def remove_outliers(self,start_time,end_time):
    ''' 
    masks data data between *start_time* and *end_time*
    '''
    xidx = self.config['xidx']
    tidx, = np.nonzero((self.t >= start_time) & (self.t <= end_time))
    self.data_sets[0][tidx,xidx] = np.nan
    self.sigma_sets[0][tidx,xidx] = np.inf
    self.log.append(('outliers',xidx,start_time,end_time))
    logger.info('removed outliers for station %s from time %s to %s' % (xidx,start_time,end_time))

  def on_mouse_press(self,event):
    # ignore if not the left button
    if event.button != 1: return
    # ignore if the event was not in an axis
    if event.inaxes is None: return
    # ignore if the event was not in the time series figure
    if not event.inaxes.figure is self.ts_fig: return
    self._mouse_is_pressed = True
    # use integer x data for consistency with the rest of PyGeoNS
    self._t1 = int(np.round(event.xdata))
    self.rects = []
    self.vlines = []
    for ax in self.ts_ax:
      ymin,ymax = ax.get_ylim()
      ax.set_ylim((ymin,ymax)) # prevent the ylims from changing after calls to draw
      r = Rectangle((self._t1,ymin),0.0,ymax-ymin,color='none',alpha=0.5,edgecolor=None)
      self.rects += [r]
      self.vlines += [ax.vlines(self._t1,ymin,ymax,color='none')]
      ax.add_patch(r)
          
    self.ts_fig.canvas.draw()  
        
  def on_mouse_move(self,event):
    # do nothing is a mouse button is not being held down
    if not self._mouse_is_pressed: return
    # ignore if the event was not in an axis
    if event.inaxes is None: return
    # ignore if the event was not in the time series figure
    if not event.inaxes.figure is self.ts_fig: return
    # use integer x data for consistency with the rest of PyGeoNS
    self._t2 = int(np.round(event.xdata))
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
    if not self._mouse_is_pressed: return
    self._mouse_is_pressed = False
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
    # _t2 needs to be set in the event that the click did not involve
    # a mouse move
    self._t2 = int(np.round(event.xdata))
    # act according to self._mode at the time of release
    if self._mode == 'OUTLIER_REMOVAL':
      mint = min(self._t1,self._t2)
      maxt = max(self._t1,self._t2)
      self.remove_outliers(mint,maxt) 
      # keep the time series axis limits fixed
      xlims = [i.get_xlim() for i in self.ts_ax]      
      ylims = [i.get_ylim() for i in self.ts_ax]      
      self.update()   
      for i,j in zip(self.ts_ax,xlims): i.set_xlim(j)
      for i,j in zip(self.ts_ax,ylims): i.set_ylim(j)
      self.ts_fig.canvas.draw()  

    elif self._mode == 'JUMP_REMOVAL':
      self.remove_jump(self._t1,abs(self._t2-self._t1))
      # keep the time series axis limits fixed
      xlims = [i.get_xlim() for i in self.ts_ax]      
      ylims = [i.get_ylim() for i in self.ts_ax]      
      self.update()   
      for i,j in zip(self.ts_ax,xlims): i.set_xlim(j)
      for i,j in zip(self.ts_ax,ylims): i.set_ylim(j)
      self.ts_fig.canvas.draw()  
    
    else: 
      return

  def _set_mode(self,name):
    if self._mode is None:
      self._mode = name
      logger.info('enabled %s mode' % name) 

  def _unset_mode(self,name):
    if self._mode is name:
      self._mode = None
      logger.info('disabled %s mode' % name) 
  
  def on_key_press(self,event):
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
      InteractiveVectorViewer.on_key_press(self,event)
        
  def on_key_release(self,event):
    if event.key == 'd':
      self._unset_mode('OUTLIER_REMOVAL')
      self.on_mouse_move(event)

    elif event.key == 'j':
      self._unset_mode('JUMP_REMOVAL')
      self.on_mouse_move(event)


def interactive_cleaner(*args,**kwargs):
  ''' 
  Runs InteractiveCleaner and returns the kept data
  '''
  ic = InteractiveCleaner(*args,**kwargs)
  ic.connect()
  return ic.get_data()
  
                                      
