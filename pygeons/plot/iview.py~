#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from rbf.interpolate import RBFInterpolant
from pygeons.plot.rin import restricted_input
from pygeons.plot.quiver import Quiver
from matplotlib.cm import ScalarMappable
from rbf.basis import phs1
import logging
import scipy.interpolate
from scipy.spatial import cKDTree
try:
  from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
except ImportError:
  from PyQt5.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
  
logger = logging.getLogger(__name__)

def _roll(lst):
  # rolls elements by 1 to the right. does not convert lst to an array
  out = [lst[-1]] + lst[:-1]
  return out


def _grid_interp_data(u,pnts,x,y):
  # u must be a masked array
  u = np.asarray(u)
  pnts = np.asarray(pnts)
  x = np.asarray(x)  
  y = np.asarray(y)

  pnts = pnts[np.isfinite(u)]
  u = u[np.isfinite(u)] 
  # return an array of zeros if all data is masked or if there is no 
  # data
  if pnts.shape[0] == 0:
    return np.zeros((x.shape[0],y.shape[0]))
        
  xg,yg = np.meshgrid(x,y)
  xf,yf = xg.flatten(),yg.flatten()
  pnts_itp = np.array([xf,yf]).T
  I = scipy.interpolate.NearestNDInterpolator(pnts,u)
  # uncomment to use a smooth interpolator
  #I = RBFInterpolant(pnts,u,penalty=0.0,
  #                   order=1,basis=phs1)
  uitp = I(pnts_itp)
  uitp = uitp.reshape((x.shape[0],y.shape[0]))                   
  return uitp
  

def one_sigfig(val):
  ''' 
  rounds *val* to one significant figure
  '''
  if np.isfinite(val):
    return np.round(val,-int(np.floor(np.log10(np.abs(val)))))
  else:
    return val  


def disable_default_key_bindings():
  ''' 
  removes the default matplotlib key bindings
  '''
  for k in plt.rcParams.keys():
    if 'keymap' in k:
      plt.rcParams[k] = []


def without_interactivity(fin):
  ''' 
  wrapper which turns figure interactivity off during the function 
  call
  '''
  def fout(*args,**kwargs):
    print('temporarily disabling figure interactivity\n')
    pyqtRemoveInputHook()
    out = fin(*args,**kwargs)
    pyqtRestoreInputHook()
    print('figure interactivity has been restored\n')
    return out

  return fout
    

class InteractiveViewer(object):
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
  def __init__(self,t,x,
               u=None,v=None,z=None,
               su=None,sv=None,sz=None, 
               units=None,
               quiver_key_length=None,
               quiver_scale=None,
               quiver_key_pos=(0.2,0.1),
               scatter_size=100,
               image_clim=None,
               image_cmap=None,
               image_array_size=200,
               station_labels=None,
               time_labels=None,
               data_set_labels=None,
               ts_ax=None,
               ts_title=None,
               fontsize=10,
               map_ax=None,
               map_title=None,
               map_ylim=None,
               map_xlim=None,
               color_cycle=None):
    ''' 
    interactively views vector valued data which is time and space 
    dependent
    
    Parameters
    ----------
      t : (Nt,) array
        observation times

      x : (Nx,2) array
        observation positions
        
      u,v,z : (S,Nx,Nt) array
        vector field components
        
      su,sv,sz : (S,Nx,Nt) array
        one standard deviation uncertainty in vector field components
        
      units : str
      
      quiver_key_length : float
        length of the quiver key

      quiver_scale : float
        scales the vectors by this amount

      quiver_key_pos : (2,) array
        position of th quiver key in axis coordinates
        
      scatter_size : float
        size of the vertical deformation dots

      station_labels : (Nx,) str array
      
      data_set_labels : (Ns,) str array
      
      image_clim : float
        minimum and maximum vertical color value      

      image_cmap : Colormap instance
        colormap for vertical deformation

      image_array_size : int
        number of columns and rows in the matrix passed to plt.imshow. 
        Larger number produces crisper Voronoi cells

      map_ax : Axis instance
        axis where map view will be plotted
      
      map_title : str
        replaces the default title for the map view plot

      map_ylim : (2,) array
        ylim for the map view plot
      
      map_xlim : (2,) array
        xlim for the map view plot
      
      ts_ax : Axis instance
        list of three axes where the time series components will be 
        plotted. They must all be on the same figure in order for 
        interactivity to work

      ts_title : str
        title for time series plot
      
      fontsize : float
        controls all fontsizes
        
    '''
    # time and space arrays
    self.t = np.asarray(t)
    self.x = np.asarray(x)
    Nt = self.t.shape[0]
    Nx = self.x.shape[0]
    
    # find out how many data sets were provided
    if u is not None:
      S = len(u)
    elif v is not None:
      S = len(v)       
    elif z is not None:
      S = len(z)       
    else:
      raise ValueError('must provide either u, v, or z')  
    
    if u is None:
      u = [np.zeros((Nt,Nx)) for i in range(S)]
    if v is None:
      v = [np.zeros((Nt,Nx)) for i in range(S)]
    if z is None:
      z = [np.zeros((Nt,Nx)) for i in range(S)]
    if su is None:
      su = [np.zeros((Nt,Nx)) for i in range(S)]
    if sv is None:
      sv = [np.zeros((Nt,Nx)) for i in range(S)]
    if sz is None:
      sz = [np.zeros((Nt,Nx)) for i in range(S)]
  
    if ((len(u)  != S) | (len(v)  != S) |
        (len(z)  != S) | (len(su) != S) |
        (len(sv) != S) | (len(sz) != S)):
      raise ValueError(
        'provided values of u, v, z, su, sv, and sz must have the '
        'same length')

    if color_cycle is None:
      self.color_cycle = ['k',(0.0,0.7,0.0),'b','r','m','c','y']
    else:
      self.color_cycle = color_cycle

    # merge u,v,z and su,sv,sz into data_sets and sigma_sets for 
    # compactness
    self.data_sets = []
    for i,j,k in zip(u,v,z):
      i,j,k = np.asarray(i),np.asarray(j),np.asarray(k)
      tpl = (i[:,:,None],j[:,:,None],k[:,:,None])
      self.data_sets += [np.concatenate(tpl,axis=2)]

    self.sigma_sets = []
    for i,j,k in zip(su,sv,sz):
      i,j,k = np.asarray(i),np.asarray(j),np.asarray(k)
      tpl = (i[:,:,None],j[:,:,None],k[:,:,None])
      self.sigma_sets += [np.concatenate(tpl,axis=2)]
    
    # map view axis and figure
    if map_ax is None:
      # gives a white background 
      map_fig,map_ax = plt.subplots(num='Map View',
                                    facecolor='white')
      self.map_fig = map_fig
      self.map_ax = map_ax
    else:
      self.map_fig = map_ax.get_figure()  
      self.map_ax = map_ax

    # make figure and axis for the time series 
    if ts_ax is None:
      ts_fig,ts_ax = plt.subplots(3,1,sharex=True,
                                  num='Time Series View',
                                  facecolor='white')
      self.ts_fig = ts_fig
      self.ts_ax = ts_ax
    else:
      self.ts_fig = ts_ax[0].get_figure()
      self.ts_ax = ts_ax   
      
    # colorbar axis
    self.cax = None
      
    # station names used for the time series plots
    if station_labels is None:
      station_labels = np.arange(len(self.x)).astype(str)

    if time_labels is None:
      time_labels = np.array(self.t).astype(str)

    # data set names used for the legends
    if data_set_labels is None:
      data_set_labels = np.arange(len(self.data_sets)).astype(str)

    self.station_labels = list(station_labels)
    self.time_labels = list(time_labels)
    self.data_set_labels = list(data_set_labels)

    if quiver_key_length is None: 
      # find the average length of unmasked data
      mags = np.linalg.norm(self.data_sets[0],axis=2)  
      mag = max(np.nanmean(mags),1e-10)
      # round to leading sigfig
      quiver_key_length = one_sigfig(mag)
      
    if quiver_scale is None:
      mags = np.linalg.norm(self.data_sets[0],axis=2)  
      mag = max(np.nanmean(mags),1e-10)
      # find the average shortest distance between points
      if Nx <= 1:
        dist = 1.0
      else:  
        T = cKDTree(self.x)
        dist = np.mean(T.query(self.x,2)[0][:,1])

      quiver_scale = mag/dist
      
    # this dictionary contains plot configuration parameters which may 
    # be interactively changed
    self.config = {}
    self.config['highlight'] = True
    self.config['tidx'] = 0
    self.config['xidx'] = 0
    self.config['units'] = units
    self.config['image_cmap'] = image_cmap
    self.config['image_clim'] = image_clim
    self.config['image_array_size'] = image_array_size
    self.config['quiver_scale'] = quiver_scale
    self.config['quiver_key_pos'] = quiver_key_pos        
    self.config['quiver_key_length'] = quiver_key_length
    self.config['scatter_size'] = scatter_size
    self.config['ts_title'] = ts_title
    self.config['map_title'] = map_title
    self.config['map_xlim'] = map_xlim
    self.config['map_ylim'] = map_ylim
    self.config['fontsize'] = fontsize
    self._init()
    disable_default_key_bindings()
    print(self.__doc__)

  def connect(self):
    self.ts_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.ts_fig.canvas.mpl_connect('pick_event',self.on_pick)
    self.map_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.map_fig.canvas.mpl_connect('pick_event',self.on_pick)

  def _init_ts_ax(self):
    # call after _init_lines
    if self.config['units'] is None:
      ts_ylabel_0 = 'east'
      ts_ylabel_1 = 'north'
      ts_ylabel_2 = 'vertical' 

    else:
      ts_ylabel_0 = 'east [%s]' % self.config['units']
      ts_ylabel_1 = 'north [%s]' % self.config['units']
      ts_ylabel_2 = 'vertical [%s]' % self.config['units']

    self.ts_ax[2].set_xlabel('time')
    self.ts_ax[0].set_ylabel(ts_ylabel_0)
    self.ts_ax[1].set_ylabel(ts_ylabel_1)
    self.ts_ax[2].set_ylabel(ts_ylabel_2)
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
      name = self.station_labels[self.config['xidx']]
      self.ts_ax[0].set_title('station : %s' % name,
                              fontsize=self.config['fontsize'])
    else:
      self.ts_ax[0].set_title(self.config['ts_title'],
                              fontsize=self.config['fontsize'])

    self.ts_ax[0].legend(frameon=False,fontsize=self.config['fontsize'])
    # hide x tick labels for the top two axes
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
      name = self.station_labels[self.config['xidx']]
      self.ts_ax[0].set_title('station : %s' % name,
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
      time_label = self.time_labels[self.config['tidx']]
      self.map_ax.set_title('time : %s' % time_label,
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
      time_label = self.time_labels[self.config['tidx']]
      self.map_ax.set_title('time : %s' % time_label,
                            fontsize=self.config['fontsize'])
    else:
      self.map_ax.set_title(self.config['map_title'],
                            fontsize=self.config['fontsize'])

  def _init_image(self):
    # call after _init_map_ax    
    self.x_itp = [np.linspace(self.config['map_xlim'][0],
                              self.config['map_xlim'][1],
                              self.config['image_array_size']),
                  np.linspace(self.config['map_ylim'][0],
                              self.config['map_ylim'][1],
                              self.config['image_array_size'])]
                              
    data_itp = _grid_interp_data(self.data_sets[0][self.config['tidx'],:,2],
                                 self.x,self.x_itp[0],self.x_itp[1])
    if self.config['image_clim'] is None:
      # if vmin and vmax are None then the color bounds will be 
      # updated each time the artists are redrawn
      image_clim = data_itp.min(),data_itp.max()
    else:  
      image_clim = self.config['image_clim']

    self.image = self.map_ax.imshow(
                   data_itp,
                   extent=(self.config['map_xlim']+self.config['map_ylim']),
                   interpolation='bicubic',
                   origin='lower',
                   vmin=image_clim[0],vmax=image_clim[1],
                   cmap=self.config['image_cmap'],
                   zorder=0)

    # make colorbar     
    # if a color bar axis has not already been made then make one
    if self.cax is None:
      self.cbar = self.map_fig.colorbar(self.image,ax=self.map_ax)  
      self.cax = self.cbar.ax
    else:
      self.cbar = self.map_fig.colorbar(self.image,cax=self.cax)  
      
    if self.config['units'] is None:
      image_clabel = 'vertical'
    else:
      image_clabel = 'vertical [%s]' % self.config['units']
      
    self.cbar.set_clim(image_clim)
    self.cbar.set_label(image_clabel,
                        fontsize=self.config['fontsize'])
    self.cbar.ax.tick_params(labelsize=self.config['fontsize'])
    self.cbar.solids.set_rasterized(True)

  def _update_image(self):
    # updates for:
    #   tidx
    #   image_clim
    data_itp = _grid_interp_data(self.data_sets[0][self.config['tidx'],:,2],
                                 self.x,
                                 self.x_itp[0],
                                 self.x_itp[1])
    self.image.set_data(data_itp)
    
    if self.config['image_clim'] is None:
      # self.image_clim are the user specified color bounds. if they 
      # are None then the color bounds will be updated each time the 
      # artists are redrawn
      image_clim = data_itp.min(),data_itp.max()
    else:  
      image_clim = self.config['image_clim']

    self.image.set_clim(image_clim)
    self.cbar.set_clim(image_clim)
    
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
                     edgecolor=self.color_cycle[1])

  def _update_scatter(self):
    # call after _update_image
    # 
    # updates for:
    #   tidx
    #   image_clim
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
      q = Quiver(self.map_ax,self.x[:,0],self.x[:,1],
                 self.data_sets[si][self.config['tidx'],:,0],
                 self.data_sets[si][self.config['tidx'],:,1],
                 scale=self.config['quiver_scale'],  
                 width=0.005,
                 sigma=(self.sigma_sets[si][self.config['tidx'],:,0],
                        self.sigma_sets[si][self.config['tidx'],:,1],
                        np.zeros(self.x.shape[0])), 
                 color=self.color_cycle[si],
                 ellipse_kwargs={'edgecolors':'k','zorder':1+si},
                 zorder=2+si)
      self.map_ax.add_collection(q,autolim=True)
      self.map_ax.autoscale_view()                 
      self.quiver += [q]                        
      if si == 0:
        # plot quiver key for the first data set
        if self.config['units'] is None:
          quiver_key_label = '%s' % self.config['quiver_key_length']
        else:
          quiver_key_label = '%s %s' % (self.config['quiver_key_length'],
                                        self.config['units'])
          
        self.key = self.map_ax.quiverkey(
                     self.quiver[si],
                     self.config['quiver_key_pos'][0],
                     self.config['quiver_key_pos'][1],
                     self.config['quiver_key_length'],
                     quiver_key_label,
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
                               np.zeros(self.x.shape[0])))

  def _init_pickers(self):
    # pickable artists
    self.pickers = []
    for xi in self.x:
      self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                       picker=10,
                                       markersize=0)

  def _init_lines(self):
    self.line1,self.line2,self.line3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.line1 += self.ts_ax[0].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],0],
                   color=self.color_cycle[si],
                   label=self.data_set_labels[si],
                   marker='.')
      self.line2 += self.ts_ax[1].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],1],
                   color=self.color_cycle[si],
                   label=self.data_set_labels[si],
                   marker='.')
      self.line3 += self.ts_ax[2].plot(
                   self.t,
                   self.data_sets[si][:,self.config['xidx'],2],
                   color=self.color_cycle[si],
                   label=self.data_set_labels[si],
                   marker='.')
    
  def _update_lines(self):
    # updates for:
    #   xidx
    for si in range(len(self.data_sets)):
      self.line1[si].set_data(self.t,
                              self.data_sets[si][:,self.config['xidx'],0])
      # relabel in case the data_set order has switched
      self.line1[si].set_label(self.data_set_labels[si])                     
      self.line2[si].set_data(self.t,
                              self.data_sets[si][:,self.config['xidx'],1])
      self.line2[si].set_label(self.data_set_labels[si])                     
      self.line3[si].set_data(self.t,
                              self.data_sets[si][:,self.config['xidx'],2])
      self.line3[si].set_label(self.data_set_labels[si])                     
  
  def _init_fill(self):
    self.fill1,self.fill2,self.fill3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.fill1 += [self.ts_ax[0].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],0] -
                    self.sigma_sets[si][:,self.config['xidx'],0],
                    self.data_sets[si][:,self.config['xidx'],0] +
                    self.sigma_sets[si][:,self.config['xidx'],0],
                    edgecolor='none',
                    color=self.color_cycle[si],alpha=0.5)]
      self.fill2 += [self.ts_ax[1].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],1] -
                    self.sigma_sets[si][:,self.config['xidx'],1],
                    self.data_sets[si][:,self.config['xidx'],1] +
                    self.sigma_sets[si][:,self.config['xidx'],1],
                    edgecolor='none',
                    color=self.color_cycle[si],alpha=0.5)]
      self.fill3 += [self.ts_ax[2].fill_between(
                    self.t,
                    self.data_sets[si][:,self.config['xidx'],2] -
                    self.sigma_sets[si][:,self.config['xidx'],2],
                    self.data_sets[si][:,self.config['xidx'],2] +
                    self.sigma_sets[si][:,self.config['xidx'],2],
                    edgecolor='none',
                    color=self.color_cycle[si],alpha=0.5)]
  
  def _update_fill(self):
    # updates for:
    #   xidx
    [f.remove() for f in self.fill1]
    [f.remove() for f in self.fill2]
    [f.remove() for f in self.fill3]
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
    #self.map_fig.tight_layout()
    self.map_fig.canvas.draw()
    self.ts_fig.tight_layout()
    self.ts_fig.canvas.draw()

  def update(self):
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


  def hard_update(self):
    # clears all axes and redraws everything
    [f.remove() for f in self.fill1]
    [f.remove() for f in self.fill2]
    [f.remove() for f in self.fill3]
    [l.remove() for l in self.line1]
    [l.remove() for l in self.line2]
    [l.remove() for l in self.line3]
    [q.remove() for q in self.quiver]
    [p.remove() for p in self.pickers]
    self.key.remove()
    self.marker.remove()
    self.image.remove()
    if self.scatter is not None:
      self.scatter.remove()        

    self.cax.clear()
    self._init()
    
  @without_interactivity
  def command_line_configure(self):
    while True:
      try:
        key = raw_input('enter parameter name ["help" for choices or "exit"] >>> ')
        print('')
        val = self.config[key]
        break
      except KeyError:  
        if key == 'exit':
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
      print('the following error was raised when evaluating the '
            'above expression:\n    %s\n' % repr(err))
      new_val = val
        
    self.config[key] = new_val
  
  def on_pick(self,event):
    for i,v in enumerate(self.pickers):
      if event.artist == v:
        self.config['xidx'] = i
        break

    self.update()    

  def on_key_press(self,event):
    if event.key == 'right':
      self.config['tidx'] += 1
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'ctrl+right':
      self.config['tidx'] += 10
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'alt+right':
      self.config['tidx'] += 100
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'left':
      self.config['tidx'] -= 1
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'ctrl+left':
      self.config['tidx'] -= 10
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'alt+left':
      self.config['tidx'] -= 100
      Nt = self.data_sets[0].shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'up':
      self.config['xidx'] += 1
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'ctrl+up':
      self.config['xidx'] += 10
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'alt+up':
      self.config['xidx'] += 100
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'down':
      self.config['xidx'] -= 1
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'ctrl+down':
      self.config['xidx'] -= 10
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'alt+down':
      self.config['xidx'] -= 100
      Nx = self.data_sets[0].shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'h':
      self.config['highlight'] = not self.config['highlight']
      self.update()

    elif event.key == 'c':
      # cycle data arrays 
      self.data_sets = _roll(self.data_sets)
      self.data_set_labels = _roll(self.data_set_labels)
      self.sigma_sets = _roll(self.sigma_sets)
      self.update()
      
    elif event.key == 'r':
      # refresh  
      self.hard_update()
      
    elif event.key == 'enter':
      self.command_line_configure()
      self.hard_update()
      
    else:
      # do nothing
      return


def interactive_viewer(*args,**kwargs):
  ''' 
  wrapper to initiate and show an InteractiveViewer
  '''
  iv = InteractiveViewer(*args,**kwargs)
  iv.connect()
  plt.show()
  return   

