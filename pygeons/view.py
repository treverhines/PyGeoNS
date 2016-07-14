#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from rbf.interpolate import RBFInterpolant
from pygeons._input import restricted_input
from pygeons.quiver import Quiver as _Quiver
from matplotlib.cm import ScalarMappable
from rbf.basis import phs1
from logging import getLogger
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
logger = getLogger(__name__)

COLOR_CYCLE = ['k',(0.0,0.7,0.0),'r','g','c','m','y']

# change behavior of mpl.quiver. this is necessary for error 
# ellipses but may lead to insidious bugs... 
matplotlib.quiver.Quiver = _Quiver

def _roll(lst):
  # rolls elements by 1 to the right. does not convert lst to an array
  out = [lst[-1]] + lst[:-1]
  return out
  
def _make_masked_array(data_set,sigma_set):
  ''' 
  returns masked array for data_set and sigma_set. The mask is true if 
  the uncertainty for any component (su,sv,sz) is np.inf.
  '''
  data_set = np.asarray(data_set)  
  sigma_set = np.asarray(sigma_set)  
  mask = np.any(np.isinf(sigma_set),axis=2)[:,None].repeat(3)
  data_set = np.ma.masked_array(data_set,mask=mask)
  sigma_set = np.ma.masked_array(sigma_set,mask=mask)
  return data_set,sigma_set


def _grid_interp_data(u,pnts,x,y):
  if np.ma.isMaskedArray(u):
    pnts = pnts[~u.mask]
    u = u[~u.mask] 

  u = np.asarray(u)
  pnts = np.asarray(pnts)
  x = np.asarray(x)  
  y = np.asarray(y)
  
  xg,yg = np.meshgrid(x,y)
  xf,yf = xg.flatten(),yg.flatten()
  pnts_itp = np.array([xf,yf]).T
  I = RBFInterpolant(pnts,u,penalty=0.0,
                     order=1,basis=phs1)
  uitp = I(pnts_itp)
  uitp = uitp.reshape((x.shape[0],y.shape[0]))                   
  return uitp
  
def _disable_default_key_bindings():
  for k in plt.rcParams.keys():
    if 'keymap' in k:
      plt.rcParams[k] = []


class InteractiveViewer:
  ''' 
               --------------------------------
               PyGeoNS Interactive Viewer (PIV)
               --------------------------------
                                 
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
        
    C : cycle the ordering of the data sets
 
    H : hide station marker
        
Notes
-----
    Stations may also be selected by clicking on them 
    
    Exit PIV by closing the figures
  
    Key bindings only work when the active window is one of the PIV 
    figures   

---------------------------------------------------------------------
  '''
  def __init__(self,data_sets,t,x,
               sigma_sets=None,
               quiver_key_label=None,
               quiver_key_length=1.0,
               quiver_scale=10.0,
               quiver_width=0.005,
               quiver_key_pos=None,
               scatter_size=100,
               image_vmin=None,
               image_vmax=None,
               image_cmap=None,
               station_names=None,
               data_set_names=None,
               ts_title=None,
               ts_ylabel_0='easting [m]',
               ts_ylabel_1='northing [m]',
               ts_ylabel_2='vertical [m]',
               ts_xlabel='time [years]',
               fontsize=10,
               map_ax=None,
               map_title=None,
               map_ylim=None,
               map_xlim=None,
               map_clabel='vertical displacement [m]'):
    ''' 
    interactively views vector valued data which is time and space 
    dependent
    
    Parameters
    ----------
      data_sets : (Ns,) list of (Nt,Nx,3) arrays
        
      t : (Nt,) array

      x : (Nx,2) array
      
      sigma_sets : (Ns,) list of (Nt,Nx,3) arrays
        if an entry is np.inf then all components for that station at 
        that time will be masked
        
      image_cmap : Colormap instance
        colormap for vertical deformation

      quiver_key_label : str
        label above the quiver key
        
      quiver_key_length : float
        length of the quiver key

      quiver_scale : float
        scales the vectors by this amount

      quiver_key_pos : (2,) array
        position of th quiver key in axis coordinates
        
      scatter_size : float
        size of the vertical deformation dots
        
      station_names : (Nx,) str array
      
      data_set_names : (Ns,) str array
      
      image_vmin : float
        minimum vertical color value      

      image_vmax : float
        mmaximum vertical color value      
              
      map_ylim : (2,) array
        ylim for the map view plot
      
      map_xlim : (2,) array
        xlim for the map view plot
      
      ts_title : str
        title for time series plot
      
      ax : Axis instance
        axis where map view will be plotted
      
      map_title : str
        replaces the default title for the map view plot
      
      fontsize : float
        
      ts_ylabel : str
        time series y label
      
      ts_xlabel : str
        time series x label
        
      map_clabel : str
        color bar label  
      
    '''
    # map view axis and figure
    if map_ax is None:
      map_fig,map_ax = plt.subplots()
      self.map_fig = map_fig
      self.map_ax = map_ax
    else:
      self.map_fig = map_ax.get_figure()  
      self.map_ax = map_ax

    # make figure and axis for the time series 
    ts_fig,ts_ax = plt.subplots(3,1,sharex=True)
    self.ts_fig = ts_fig
    self.ts_ax = ts_ax
      
    # colorbar axis
    self.cax = None
    
    # time and space arrays
    self.t = t
    self.x = x

    # use uncertainties of zero if none are given
    if sigma_sets is None:
      sigma_sets = [np.zeros(d.shape) for d in data_sets]

    # convert data and uncertainty to masked arrays
    self.data_sets = []
    self.sigma_sets = []
    for d,s in zip(data_sets,sigma_sets):      
      dout,sout = _make_masked_array(d,s)
      self.data_sets += [dout]
      self.sigma_sets += [sout]
      
    # station names used for the time series plots
    if station_names is None:
      station_names = np.arange(len(self.x)).astype(str)

    # data set names used for the legends
    if data_set_names is None:
      data_set_names = np.arange(len(self.data_sets)).astype(str)

    self.station_names = list(station_names)
    self.data_set_names = list(data_set_names)


    # position and length of the scale vector 
    if quiver_key_pos is None:
      quiver_key_pos = (0.2,0.1)

    if quiver_key_label is None:   
      quiver_key_label = str(quiver_key_length) + ' [m]'

    # this dictionary contains plot configuration parameters which may 
    # be interactively changed
    self.config = {}
    self.config['highlight'] = True
    self.config['tidx'] = 0
    self.config['xidx'] = 0
    self.config['image_cmap'] = image_cmap
    self.config['image_vmin'] = image_vmin        
    self.config['image_vmax'] = image_vmax
    self.config['quiver_scale'] = quiver_scale
    self.config['quiver_width'] = quiver_width
    self.config['quiver_key_pos'] = quiver_key_pos        
    self.config['quiver_key_label'] = quiver_key_label
    self.config['quiver_key_length'] = quiver_key_length
    self.config['scatter_size'] = scatter_size
    self.config['ts_xlabel'] = ts_xlabel
    self.config['ts_ylabel_0'] = ts_ylabel_0
    self.config['ts_ylabel_1'] = ts_ylabel_1
    self.config['ts_ylabel_2'] = ts_ylabel_2
    self.config['ts_title'] = ts_title
    self.config['map_title'] = map_title
    self.config['map_clabel'] = map_clabel
    self.config['map_xlim'] = map_xlim
    self.config['map_ylim'] = map_ylim
    self.config['fontsize'] = fontsize
    self._init()
    _disable_default_key_bindings()
    print(self.__doc__)

  def connect(self):
    self.ts_fig.canvas.mpl_connect('key_press_event',self._onkey)
    self.ts_fig.canvas.mpl_connect('pick_event',self._onpick)
    self.map_fig.canvas.mpl_connect('key_press_event',self._onkey)
    self.map_fig.canvas.mpl_connect('pick_event',self._onpick)

  def _init_ts_ax(self):
    # call after _init_lines
    self.ts_ax[2].set_xlabel(self.config['ts_xlabel'])
    self.ts_ax[0].set_ylabel(self.config['ts_ylabel_0'])
    self.ts_ax[1].set_ylabel(self.config['ts_ylabel_1'])
    self.ts_ax[2].set_ylabel(self.config['ts_ylabel_2'])
    # dont convert to exponential form
    self.ts_ax[0].get_xaxis().get_major_formatter().set_useOffset(False)
    self.ts_ax[1].get_xaxis().get_major_formatter().set_useOffset(False)
    self.ts_ax[2].get_xaxis().get_major_formatter().set_useOffset(False)
    self.ts_ax[0].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[1].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[2].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[0].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[1].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[2].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[0].title.set_fontsize(self.config['fontsize'])
    self.ts_ax[0].tick_params(labelsize=self.config['fontsize'])
    self.ts_ax[1].tick_params(labelsize=self.config['fontsize'])
    self.ts_ax[2].tick_params(labelsize=self.config['fontsize'])
    if self.config['ts_title'] is None:
      name = self.station_names[self.config['xidx']]
      self.ts_ax[0].set_title('station %s' % name,
                              fontsize=self.config['fontsize'])
    else:
      self.ts_ax[0].set_title(self.config['ts_title'],
                              fontsize=self.config['fontsize'])

    self.ts_ax[0].legend(frameon=False,fontsize=self.config['fontsize'])
    plt.setp(self.ts_ax[0].get_xticklabels(), visible=False)
    plt.setp(self.ts_ax[1].get_xticklabels(), visible=False)
    self.ts_ax[0].set_autoscale_on(True) 
    self.ts_ax[1].set_autoscale_on(True) 
    self.ts_ax[2].set_autoscale_on(True) 
    self.ts_ax[0].relim()
    self.ts_ax[1].relim()
    self.ts_ax[2].relim()
    self.ts_ax[0].autoscale_view()
    self.ts_ax[1].autoscale_view()
    self.ts_ax[2].autoscale_view()
    
  def _update_ts_ax(self):
    # call after _update_lines
    #
    # updates for:
    #   xidx
    #   ts_title
    if self.config['ts_title'] is None:
      name = self.station_names[self.config['xidx']]
      self.ts_ax[0].set_title('station %s' % name,
                              fontsize=self.config['fontsize'])
    else:
      self.ts_ax[0].set_title(self.config['ts_title'],
                              fontsize=self.config['fontsize'])

    self.ts_ax[0].legend(frameon=False,fontsize=self.config['fontsize'])
    self.ts_ax[0].set_autoscale_on(True) 
    self.ts_ax[1].set_autoscale_on(True) 
    self.ts_ax[2].set_autoscale_on(True) 
    self.ts_ax[0].relim()
    self.ts_ax[1].relim()
    self.ts_ax[2].relim()
    self.ts_ax[0].autoscale_view()
    self.ts_ax[1].autoscale_view()
    self.ts_ax[2].autoscale_view()
    

  def _init_map_ax(self): 
    # call after _init_scatter
    self.map_ax.set_aspect('equal')
    self.map_ax.tick_params(labelsize=self.config['fontsize'])
    if self.config['map_title'] is None:
      time = self.t[self.config['tidx']]
      self.map_ax.set_title('time : %g' % time,
                            fontsize=self.config['fontsize'])
    else:
      self.map_ax.set_title(self.config['map_title'],
                            fontsize=self.config['fontsize'])

    # do not dynamically update the axis limits
    if self.config['map_xlim'] is None:
      self.config['map_xlim'] = self.map_ax.get_xlim()

    if self.config['map_ylim'] is None:  
      self.config['map_ylim'] = self.map_ax.get_ylim()
      
    self.map_ax.set_xlim(self.config['map_xlim'])
    self.map_ax.set_ylim(self.config['map_ylim'])
      
  def _update_map_ax(self):
    # updates for:
    #   map_title
    #   tidx
    if self.config['map_title'] is None:
      time = self.t[self.config['tidx']]
      self.map_ax.set_title('time : %g' % time,
                            fontsize=self.config['fontsize'])
    else:
      self.map_ax.set_title(self.config['map_title'],
                            fontsize=self.config['fontsize'])

  def _init_image(self):
    # call after _init_map_ax    
    self.x_itp = [np.linspace(self.config['map_xlim'][0],self.config['map_xlim'][1],100),
                  np.linspace(self.config['map_ylim'][0],self.config['map_ylim'][1],100)]
    data_itp = _grid_interp_data(self.data_sets[0][self.config['tidx'],:,2],
                                 self.x,self.x_itp[0],self.x_itp[1])
    if self.config['image_vmin'] is None:
      # if vmin and vmax are None then the color bounds will be 
      # updated each time the artists are redrawn
      image_vmin = data_itp.min()
    else:  
      image_vmin = self.config['image_vmin']

    if self.config['image_vmax'] is None:
      image_vmax = data_itp.max()
    else:
      image_vmax = self.config['image_vmax']
          
    self.image = self.map_ax.imshow(
                   data_itp,
                   extent=(self.config['map_xlim']+self.config['map_ylim']),
                   interpolation='bicubic',
                   origin='lower',
                   vmin=image_vmin,vmax=image_vmax,
                   cmap=self.config['image_cmap'],
                   zorder=0)

    # make colorbar     
    self.cbar = self.map_fig.colorbar(self.image,cax=self.cax)  
    self.cax = self.cbar.ax
    self.cbar.set_clim((image_vmin,image_vmax))
    self.cbar.set_label(self.config['map_clabel'],
                        fontsize=self.config['fontsize'])
    self.cbar.ax.tick_params(labelsize=self.config['fontsize'])
    self.cbar.solids.set_rasterized(True)

  def _update_image(self):
    # updates for:
    #   tidx
    #   image_vmin
    #   image_vmax  
    data_itp = _grid_interp_data(self.data_sets[0][self.config['tidx'],:,2],
                                 self.x,
                                 self.x_itp[0],
                                 self.x_itp[1])
    self.image.set_data(data_itp)
    
    if self.config['image_vmin'] is None:
      # self.image_vmin and self.image_vmax are the user specified color 
      # bounds. if they are None then the color bounds will be 
      # updated each time the artists are redrawn
      image_vmin = data_itp.min()
    else:  
      image_vmin = self.config['image_vmin']

    if self.config['image_vmax'] is None:
      image_vmax = data_itp.max()
    else:
      image_vmax = self.config['image_vmax']

    self.image.set_clim((image_vmin,image_vmax))
    self.cbar.set_clim((image_vmin,image_vmax))
    self.cbar.set_label(self.config['map_clabel'],
                        fontsize=self.config['fontsize'])
    self.cbar.ax.tick_params(labelsize=self.config['fontsize'])
    self.cbar.solids.set_rasterized(True)
    
  def _init_scatter(self):
    # call after _init_image
    if len(self.data_sets) < 2:
      self.scatter = None
      return

    sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cbar.get_cmap())
    # use scatter points to show z for second data set 
    colors = sm.to_rgba(self.data_sets[1][self.config['tidx'],:,2])
    self.scatter = self.map_ax.scatter(
                     self.x[:,0],self.x[:,1],
                     c=colors,
                     s=self.config['scatter_size'],
                     zorder=1,
                     edgecolor=COLOR_CYCLE[1])

  def _update_scatter(self):
    # call after _update_image
    # 
    # updates for:
    #   tidx
    #   image_vmin
    #   image_vmax
    if len(self.data_sets) < 2:
      return

    sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cbar.get_cmap())
    colors = sm.to_rgba(self.data_sets[1][self.config['tidx'],:,2])
    self.scatter.set_facecolors(colors)

  def _init_marker(self):
    self.marker = self.map_ax.plot(self.x[self.config['xidx'],0],
                                   self.x[self.config['xidx'],1],'ko',
                                   markersize=20*self.config['highlight'])[0]

  def _update_marker(self):
    # updates for:
    #   xidx
    #   highlight
    self.marker.set_data(self.x[self.config['xidx'],0],
                         self.x[self.config['xidx'],1])
    self.marker.set_markersize(20*self.config['highlight'])

  def _init_quiver(self):
    self.quiver = []
    for si in range(len(self.data_sets)):
      self.quiver += [self.map_ax.quiver(
                        self.x[:,0],self.x[:,1],
                        self.data_sets[si][self.config['tidx'],:,0],
                        self.data_sets[si][self.config['tidx'],:,1],
                        scale=self.config['quiver_scale'],  
                        width=self.config['quiver_width'],
                        sigma=(self.sigma_sets[si][self.config['tidx'],:,0],
                               self.sigma_sets[si][self.config['tidx'],:,1],
                               0.0*self.sigma_sets[si][self.config['tidx'],:,0]),
                        color=COLOR_CYCLE[si],
                        ellipse_kwargs={'edgecolors':'k','zorder':1+si},
                        zorder=2+si)]
      if si == 0:
        # plot quiver key for the first data set
        self.key = self.map_ax.quiverkey(
                     self.quiver[si],
                     self.config['quiver_key_pos'][0],
                     self.config['quiver_key_pos'][1],
                     self.config['quiver_key_length'],
                     self.config['quiver_key_label'],
                     zorder=2,
                     labelsep=0.05,
                     fontproperties={'size':self.config['fontsize']})
                     
  def _update_quiver(self):
    # updates for:
    #   tidx
    for si in range(len(self.data_sets)):
      self.quiver[si].set_UVC(
                        self.data_sets[si][self.config['tidx'],:,0],
                        self.data_sets[si][self.config['tidx'],:,1],
                        sigma=(self.sigma_sets[si][self.config['tidx'],:,0],
                               self.sigma_sets[si][self.config['tidx'],:,1],
                               0.0*self.sigma_sets[si][self.config['tidx'],:,0]))

  def _init_pickers(self):
    # pickable artists
    self.pickers = []
    for xi in self.x:
      self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                       picker=10,
                                       markersize=0)

  def _init_lines(self):
    self.L1,self.L2,self.L3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.L1 += self.ts_ax[0].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],0],
                   color=COLOR_CYCLE[si],
                   label=self.data_set_names[si])
      self.L2 += self.ts_ax[1].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],1],
                   color=COLOR_CYCLE[si],
                   label=self.data_set_names[si])
      self.L3 += self.ts_ax[2].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],2],
                   color=COLOR_CYCLE[si],
                   label=self.data_set_names[si])
    
  def _update_lines(self):
    # updates for:
    #   xidx
    for si in range(len(self.data_sets)):
      self.L1[si].set_data(self.t,
                           self.data_sets[si][:,self.config['xidx'],0])
      # relabel in case the data_set order has switched
      self.L1[si].set_label(self.data_set_names[si])                     
      self.L2[si].set_data(self.t,
                           self.data_sets[si][:,self.config['xidx'],1])
      self.L2[si].set_label(self.data_set_names[si])                     
      self.L3[si].set_data(self.t,
                           self.data_sets[si][:,self.config['xidx'],2])
      self.L3[si].set_label(self.data_set_names[si])                     
  
  def _init_fill(self):
    self.F1,self.F2,self.F3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.F1 += [self.ts_ax[0].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],0] -
                    self.sigma_sets[si][:,self.config['xidx'],0],
                    self.data_sets[si][:,self.config['xidx'],0] +
                    self.sigma_sets[si][:,self.config['xidx'],0],
                    edgecolor='none',
                    color=COLOR_CYCLE[si],alpha=0.5)]
      self.F2 += [self.ts_ax[1].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],1] -
                    self.sigma_sets[si][:,self.config['xidx'],1],
                    self.data_sets[si][:,self.config['xidx'],1] +
                    self.sigma_sets[si][:,self.config['xidx'],1],
                    edgecolor='none',
                    color=COLOR_CYCLE[si],alpha=0.5)]
      self.F3 += [self.ts_ax[2].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],2] -
                    self.sigma_sets[si][:,self.config['xidx'],2],
                    self.data_sets[si][:,self.config['xidx'],2] +
                    self.sigma_sets[si][:,self.config['xidx'],2],
                    edgecolor='none',
                    color=COLOR_CYCLE[si],alpha=0.5)]
  
  def _update_fill(self):
    # updates for:
    #   xidx
    [f.remove() for f in self.F1]
    [f.remove() for f in self.F2]
    [f.remove() for f in self.F3]
    self._init_fill()

  def _init(self):
    self._init_marker()
    self._init_pickers()
    self._init_map_ax()
    self._init_quiver()
    self._init_lines()
    self._init_fill()
    self._init_image()
    self._init_scatter()
    self._init_ts_ax()
    self.map_fig.tight_layout()
    self.map_fig.canvas.draw()
    self.ts_fig.tight_layout()
    self.ts_fig.canvas.draw()

  def _update(self):
    self._update_marker()
    self._update_map_ax()
    self._update_quiver()
    self._update_lines()
    self._update_fill()
    self._update_image()
    self._update_scatter()
    self._update_ts_ax()
    self.ts_fig.canvas.draw()
    self.map_fig.canvas.draw()


  def _hard_update(self):
    # clears all axes and redraws everything
    logger.debug('refreshing figures')
    [f.remove() for f in self.F1]
    [f.remove() for f in self.F2]
    [f.remove() for f in self.F3]
    [l.remove() for l in self.L1]
    [l.remove() for l in self.L2]
    [l.remove() for l in self.L3]
    [q.remove() for q in self.quiver]
    self.key.remove()
    self.marker.remove()
    self.image.remove()
    if self.scatter is not None:
      self.scatter.remove()        

    self.cax.clear()
    self._init()
    

  def _onpick(self,event):
    logger.debug('detected pick event')
    for i,v in enumerate(self.pickers):
      if event.artist == v:
        self.config['xidx'] = i
        break

    self._update()    


  def _onkey(self,event):
    logger.debug('detected key event')
    hard_update = False

    if event.key == 'right':
      self.config['tidx'] += 1

    elif event.key == 'ctrl+right':
      self.config['tidx'] += 10

    elif event.key == 'alt+right':
      self.config['tidx'] += 100

    elif event.key == 'left':
      self.config['tidx'] -= 1

    elif event.key == 'ctrl+left':
      self.config['tidx'] -= 10

    elif event.key == 'alt+left':
      self.config['tidx'] -= 100

    elif event.key == 'up':
      self.config['xidx'] += 1

    elif event.key == 'ctrl+up':
      self.config['xidx'] += 10

    elif event.key == 'alt+up':
      self.config['xidx'] += 100

    elif event.key == 'down':
      self.config['xidx'] -= 1

    elif event.key == 'ctrl+down':
      self.config['xidx'] -= 10

    elif event.key == 'alt+down':
      self.config['xidx'] -= 100

    elif event.key == 'h':
      self.config['highlight'] = not self.config['highlight']

    elif event.key == 'c':
      # cycle data arrays 
      self.data_sets = _roll(self.data_sets)
      self.data_set_names = _roll(self.data_set_names)
      self.sigma_sets = _roll(self.sigma_sets)
      
    elif event.key == 'r':
      # refresh  
      hard_update = True
      pass
      
    elif event.key == 'enter':
      hard_update = True
      print('temporarily disabling figure interactivity to allow for command line configuration\n')
      pyqtRemoveInputHook()
      while True:
        try:
          key = raw_input('enter parameter name ["help" for choices or "exit"] >>> ')
          print('')
          val = self.config[key]
          break
        except KeyError:  
          if key == 'exit':
            pyqtRestoreInputHook()
            print('figure interactivity has been restored\n')
            return
          
          if key != 'help':
            print('"%s" is not a configurable parameter\n' % key)

          print('select from one of the following')
          for k in self.config.keys():
            print('    %s' % k)
          print('')

      print('current value is %s\n' % repr(val))
      try:
        new_val = restricted_input('new value >>> ')
        print('')
        
      except Exception as err:
        print('')
        print('the following error was raised when evaluating the above expression:\n    %s\n' % repr(err))
        new_val = val
        
      self.config[key] = new_val
      pyqtRestoreInputHook()
      print('figure interactivity has been restored\n')
      
    else:
      # do nothing
      return

    self.config['tidx'] = self.config['tidx']%self.data_sets[0].shape[0]
    self.config['xidx'] = self.config['xidx']%self.data_sets[0].shape[1]
    
    if hard_update:
      self._hard_update()    
    else:  
      self._update()    
    

def network_viewer(t,x,u=None,v=None,z=None,
                   su=None,sv=None,sz=None,
                   **kwargs):
  ''' 
  makes an interactive plot of a three-component vector field which is 
  a function of time and two-dimensional space.  Produces two figures, one
  is a map view of the vector field at some time, the other is a time series 
  of the vector components for some position.   
  
  Parameters
  ----------
    t : (Nt,) array

    x : (Nx,2) array
    
    u,v,z : (Ns,) list of (Nt,Nx) arrays
      vector components all value must be finite
    
    su,sv,sz : (Ns,) list of (Nt,Nx) array
      uncertainties in u,v, and z. data with uncertainties of np.inf 
      will be treated as masked data. using zero effectively hides any 
      error ellipses or uncertainty intervals
    
    **kwargs : arguments passed to InteractiveViewer  
  
  Usage
  -----
    Interaction is done entirely with the map view figure

      right : move forward 1 time step
      ctrl-right : move forward 10 time step
      alt-right : move forward 100 time step

      right : move back 1 time step
      ctrl-right : move back 10 time step
      alt-right : move back 100 time step

      up : move up 1 station
      ctrl-up : move up 10 station
      alt-up : move up 100 station

      down : move down 1 station
      ctrl-down : move down 10 station
      alt-down : move down 100 station
      
      c : hide/reveal station marker
      
      r : rotate data_sets      
  
  Example
  -------
    >>> t = np.linspace(0,1,100) # form observation times
    >>> x = np.random.random((20,2)) # form observation positions
    >>> u,v,z = np.cumsum(np.random.normal(0.0,0.1,(3,100,20)),axis=1)
    >>> network_viewer(t,x,u=[u],v=[v],z=[z])    
  '''
  x = np.asarray(x)
  t = np.asarray(t)
  Nx = x.shape[0]
  Nt = t.shape[0]
  # find the number of data sets
  if u is not None:
    Ns = len(u)
  elif v is not None:
    Ns = len(v)
  elif z is not None:
    Ns = len(z)
  else:
    raise ValueError('one of u,v, or z must be specified')  
      
  if u is None:
    u = Ns*[np.zeros((Nt,Nx))]
  if v is None:
    v = Ns*[np.zeros((Nt,Nx))]
  if z is None:
    z = Ns*[np.zeros((Nt,Nx))]
  if su is None:
    su = Ns*[np.zeros((Nt,Nx))]
  if sv is None:
    sv = Ns*[np.zeros((Nt,Nx))]
  if sz is None:
    sz = Ns*[np.zeros((Nt,Nx))]
    
  u = [np.asarray(i) for i in u]
  v = [np.asarray(i) for i in v]
  z = [np.asarray(i) for i in z]
  su = [np.asarray(i) for i in su]
  sv = [np.asarray(i) for i in sv]
  sz = [np.asarray(i) for i in sz]
  
  if ((not all([np.isfinite(i).all() for i in u])) |
      (not all([np.isfinite(i).all() for i in v])) |
      (not all([np.isfinite(i).all() for i in z]))):
    raise ValueError('u, v, and z must all have finite values')
     
  if (any([np.isnan(i).any() for i in su]) |
      any([np.isnan(i).any() for i in sv]) |
      any([np.isnan(i).any() for i in sz])):
    raise ValueError('su, sv, and sz cannot be nan. Mask data by setting uncertainty to inf')
  
  if ((len(u) != Ns) | (len(v) != Ns) |
      (len(z) != Ns) | (len(su) != Ns) |
      (len(sv) != Ns) | (len(sz) != Ns)):
    raise ValueError('provided values of u, v, z, su, sv, or sz must have the same length')
      
  data_sets = []
  sigma_sets = []
  for i in range(Ns):
    data_sets += [np.concatenate((u[i][:,:,None],v[i][:,:,None],z[i][:,:,None]),axis=2)]
    sigma_sets += [np.concatenate((su[i][:,:,None],sv[i][:,:,None],sz[i][:,:,None]),axis=2)]
  
  iv = InteractiveViewer(data_sets,t,x,sigma_sets=sigma_sets,**kwargs)
  iv.connect()
  plt.show()
    
