import matplotlib.pyplot as plt
import numpy as np
from pygeons.plot.quiver import Quiver
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from code import InteractiveConsole
import logging
import sys
import scipy.interpolate
from scipy.spatial import cKDTree
from textwrap import wrap
try:
  from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
except ImportError:
  from PyQt5.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
  
logger = logging.getLogger(__name__)

# this is an entirely white colormap. This is used to hide the 
# vertical deformation.
_blank_cmap = ListedColormap([[1.0,1.0,1.0,0.0]])

def _roll(lst):
  # rolls elements by 1 to the right. does not convert lst to an array
  lst = list(lst)
  out = [lst[-1]] + lst[:-1]
  return out


def _grid_interp_data(u,pnts,x,y):
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
    logger.info('Temporarily disabling figure interactivity')
    pyqtRemoveInputHook()
    out = fin(*args,**kwargs)
    pyqtRestoreInputHook()
    logger.info('Figure interactivity has been restored')
    return out

  return fout
    

class Configurable(object):
  ''' 
  Class which allows the user to interactively configure its
  attributes with the *configure* method
  '''
  def configure(self,attrs=None):
    # *attrs* is a list of attribute names that the user wants to
    # configure. This defaults to all the non-private names
    if attrs is None:
      attrs = [n for n in dir(self) if not n.startswith('_')]

    # set message displayed when the interactive console starts
    msg1 = ('Python ' + sys.version + ' on ' + sys.platform + '\n'
            'Type "help", "copyright", "credits" or "license" for '
            'more information.')
    msg2 = ('The current namespace has been populated with the '
            'following attributes from the class instance:')
    msg3 = ', '.join(['"%s"' % i for i in attrs])
    msg4 = ('The values associate with these names can be modified, '
            'and the attributes of the class instance will be '
            'updated with the new values. When finished, exit the '
            'interpreter with Ctrl+"d".')

    banner  = '\n'
    banner += msg1 
    banner += '\n\n'
    banner += '\n'.join(wrap(msg2,70))
    banner += '\n\n'
    banner += '\n    '.join(wrap('    '+msg3,66))
    banner += '\n\n'    
    banner += '\n'.join(wrap(msg4,70))
    banner += '\n'    
    namespace = {}
    
    for n in attrs:
      # fill the namespace with deep copies of the classes current
      # attributes
      namespace[n] = getattr(self,n)

    # set the attribute values with an interactive console
    ic = InteractiveConsole(locals=namespace)
    ic.interact(banner=banner)
    # namespace has been modified in the interactive console. Give the
    # instance the new values
    for n in attrs:
      setattr(self,n,namespace[n])


class InteractiveVectorViewer(Configurable):
  ''' 
-------------- PyGeoNS Interactive Vector Viewer (PIVV) --------------

An interactive figure for viewing and comparing the spatial and
temporal patterns in datasets.

Controls :
  Left : Move back 1 time step (Ctrl-Left and Alt-Left move back 10
    and 100 respectively)

  Right : Move forward 1 time step (Ctrl-Right and Alt-Right move
    forward 10 and 100 respectively)

  Up : Move forward 1 station (Ctrl-Left and Alt-Left move back 10 and
    100 respectively)
          
  Down : Move back 1 station (Ctrl-Right and Alt-Right move forward 10
    and 100 respectively)
          
  R : Redraw figures
        
  C : Cycle the ordering of the datasets

  V : Toggles whether to hide the vertical component of deformation.

  H : Toggles whether to hide the station highlighter
        
  Enter : Edit parameters through the command line

Notes :
  Stations may also be selected by clicking on them.
    
  Exit PIVV by closing the figures.
  
  Key bindings only work when the active window is one of the PIVV
  figures.

----------------------------------------------------------------------
  '''
  @property
  def xidx(self):
    return self._xidx
  
  @xidx.setter
  def xidx(self,val):
    Nx = len(self.x)
    # wrap the index around so that it is between 0 and Nx
    self._xidx = val%Nx

  @property
  def tidx(self):
    return self._tidx
  
  @tidx.setter
  def tidx(self,val):
    Nt = len(self.t)
    # wrap the index around so that it is between 0 and Nt
    self._tidx = val%Nt

  def __init__(self,t,x,
               u=None,v=None,z=None,
               su=None,sv=None,sz=None, 
               station_labels=None,
               time_labels=None,
               dataset_labels=None,
               quiver_key_length=None,
               quiver_scale=None,
               quiver_key_pos=(0.15,0.2),
               image_clim=None,
               image_cmap='RdBu_r',
               image_resolution=300,
               map_ax=None,
               map_title=None,
               map_ylim=None,
               map_xlim=None,
               ts_ax=None,
               ts_title=None,
               units=None,
               scatter_size=100,
               fontsize=10,
               colors=None,
               line_styles=None,
               line_markers=None,
               error_styles=None):
    ''' 
    Interactively views vector valued data which is time and space 
    dependent
    
    Parameters
    ----------
      t : (Nt,) array
      x : (Nx,2) array
      u,v,z : (Ns,Nx,Nt) array
      su,sv,sz : (Ns,Nx,Nt) array
      **kwargs
    '''
    # SET T and X
    #################################################################
    # time and space arrays
    self.t = np.asarray(t)
    self.x = np.asarray(x)
    Nt = self.t.shape[0]
    Nx = self.x.shape[0]
    
    # SET DATA_SETS AND SIGMA_SETS
    #################################################################
    # find out how many data sets were provided
    if u is not None:
      Ns = len(u)
    elif v is not None:
      Ns = len(v)       
    elif z is not None:
      Ns = len(z)       
    else:
      raise ValueError('must provide either u, v, or z')  
    
    if u is None:
      u = [np.zeros((Nt,Nx)) for i in range(Ns)]
    if v is None:
      v = [np.zeros((Nt,Nx)) for i in range(Ns)]
    if z is None:
      z = [np.zeros((Nt,Nx)) for i in range(Ns)]
    if su is None:
      su = [np.zeros((Nt,Nx)) for i in range(Ns)]
    if sv is None:
      sv = [np.zeros((Nt,Nx)) for i in range(Ns)]
    if sz is None:
      sz = [np.zeros((Nt,Nx)) for i in range(Ns)]
  
    if ((len(u)  != Ns) | (len(v)  != Ns) |
        (len(z)  != Ns) | (len(su) != Ns) |
        (len(sv) != Ns) | (len(sz) != Ns)):
      raise ValueError(
        'provided values of u, v, z, su, sv, and sz must have the '
        'same length')

    # merge u,v,z and su,sv,sz into data_sets and sigma_sets for 
    # compactness
    self.data_sets = []
    for ui,vi,zi in zip(u,v,z):
      ui,vi,zi = np.asarray(ui),np.asarray(vi),np.asarray(zi)
      tpl = (ui[:,:,None],vi[:,:,None],zi[:,:,None])
      self.data_sets += [np.concatenate(tpl,axis=2)]

    self.sigma_sets = []
    for sui,svi,szi in zip(su,sv,sz):
      sui,svi,szi = np.asarray(sui),np.asarray(svi),np.asarray(szi)
      tpl = (sui[:,:,None],svi[:,:,None],szi[:,:,None])
      self.sigma_sets += [np.concatenate(tpl,axis=2)]
      
    ## SET MAP_AX, TS_AX, MAP_FIG, AND TS_FIG
    #################################################################
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

    ## SET PLOTTING PARAMETERS
    #################################################################
    # dataset colors
    if colors is None:
      cycle = ['k',(0.0,0.7,0.0),'b','r','m','c','y']
      colors = [cycle[i%7] for i in range(Ns)]

    # if only one color was specified then use it for all datasets
    elif len(colors) == 1:
      colors = [colors[0] for i in range(Ns)]
      
    self.colors = colors
    
    # error style
    if error_styles is None:
      error_styles = ['fill' for i in range(Ns)]

    elif len(error_styles) == 1:
      error_styles = [error_styles[0] for i in range(Ns)]
      
    self.error_styles = error_styles
    
    # line styles
    if line_styles is None:
      line_styles = ['solid' for i in range(Ns)]

    elif len(line_styles) == 1:
      line_styles = [line_styles[0] for i in range(Ns)]
      
    self.line_styles = line_styles
    
    # line markers
    if line_markers is None:
      line_markers = ['None' for i in range(Ns)]

    elif len(line_markers) == 1:
      line_markers = [line_markers[0] for i in range(Ns)]
      
    self.line_markers = line_markers

    # station labels
    if station_labels is None:
      station_labels = ['%04d' % i for i in range(Nx)]

    self.station_labels = station_labels

    # time labels
    if time_labels is None:
      time_labels = [str(i) for i in self.t]
  
    self.time_labels = time_labels

    # dataset labels
    if dataset_labels is None:
      dataset_labels = ['dataset %s' % i for i in range(Ns)]
    
    self.dataset_labels = dataset_labels  
    
    # quiver key length
    if quiver_key_length is None: 
      # This is an Nt,Nx,Ns array of vector lengths for each data set
      mags = [np.linalg.norm(d,axis=2) for d in self.data_sets]
      # find the average length of unmasked vectors
      mag = max(np.nanmean(mags),1e-10)
      # round to leading sigfig
      quiver_key_length = one_sigfig(mag)
    
    self.quiver_key_length = quiver_key_length  
      
    # quiver scale
    if quiver_scale is None:
      mags = [np.linalg.norm(d,axis=2) for d in self.data_sets]
      mag = max(np.nanmean(mags),1e-10)
      # find the average shortest distance between points
      if len(self.x) <= 1:
        dist = 1.0
      else:  
        T = cKDTree(self.x)
        dist = np.mean(T.query(self.x,2)[0][:,1])

      quiver_scale = mag/dist
    
    self.quiver_scale = quiver_scale
      
    # set additional parameters which do not need to be checked yet
    self.highlight = True
    self.tidx = 0
    self.xidx = 0
    self.units = units
    self.image_cmap = image_cmap
    self.image_clim = image_clim
    self.image_resolution = image_resolution
    self.quiver_key_pos = quiver_key_pos        
    self.scatter_size = scatter_size
    self.ts_title = ts_title
    self.map_title = map_title
    self.map_xlim = map_xlim
    self.map_ylim = map_ylim
    self.fontsize = fontsize

    # initiate all artists
    self._init()
    # turn off MPLs key bindings and use my own
    disable_default_key_bindings()
    # display help
    logger.info(self.__doc__)

  def connect(self):
    self.ts_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.ts_fig.canvas.mpl_connect('pick_event',self.on_pick)
    self.map_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.map_fig.canvas.mpl_connect('pick_event',self.on_pick)
    plt.show()

  def _init_ts_ax(self):
    # Initially configures the time series axes. This involves setting 
    # titles, labels, and scaling to fit the displayed data
    #
    # CALL THIS AFTER *_init_lines*
    #
    if self.units is None:
      ts_ylabel_0 = 'east'
      ts_ylabel_1 = 'north'
      ts_ylabel_2 = 'vertical' 

    else:
      ts_ylabel_0 = 'east [%s]' % self.units
      ts_ylabel_1 = 'north [%s]' % self.units
      ts_ylabel_2 = 'vertical [%s]' % self.units

    self.ts_ax[2].set_xlabel('time')
    self.ts_ax[0].set_ylabel(ts_ylabel_0)
    self.ts_ax[1].set_ylabel(ts_ylabel_1)
    self.ts_ax[2].set_ylabel(ts_ylabel_2)
    self.ts_ax[0].grid(zorder=0)
    self.ts_ax[1].grid(zorder=0)
    self.ts_ax[2].grid(zorder=0)
    self.ts_ax[0].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[1].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[2].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[0].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[1].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[2].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[0].title.set_fontsize(self.fontsize)
    self.ts_ax[0].tick_params(labelsize=self.fontsize)
    self.ts_ax[1].tick_params(labelsize=self.fontsize)
    self.ts_ax[2].tick_params(labelsize=self.fontsize)
    if self.ts_title is None:
      name = self.station_labels[self.xidx]
      self.ts_ax[0].set_title('station : %s' % name,
                              fontsize=self.fontsize)
    else:
      self.ts_ax[0].set_title(self.ts_title,
                              fontsize=self.fontsize)

    self.ts_ax[0].legend(frameon=False,fontsize=self.fontsize)
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
    # Updates the time series axes for changes in *tidx* or *xidx*. 
    # This involves changing the axes titles and rescaling for the new 
    # data being displayed. 
    #
    # CALL THIS AFTER *_update_lines*
    # 
    if self.ts_title is None:
      name = self.station_labels[self.xidx]
      self.ts_ax[0].set_title('station : %s' % name,
                              fontsize=self.fontsize)
    else:
      self.ts_ax[0].set_title(self.ts_title,
                              fontsize=self.fontsize)

    self.ts_ax[0].legend(frameon=False,fontsize=self.fontsize)
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
    # Initially configures the map view axis. This involves setting 
    # titles, labels, and scaling to fit the plotted data 
    #
    # CALL THIS AFTER *_init_scatter*
    # 
    self.map_ax.set_aspect('equal')
    self.map_ax.tick_params(labelsize=self.fontsize)
    if self.map_title is None:
      time_label = self.time_labels[self.tidx]
      self.map_ax.set_title('time : %s' % time_label,
                            fontsize=self.fontsize)
    else:
      self.map_ax.set_title(self.map_title,
                            fontsize=self.fontsize)

    # do not dynamically update the axis limits
    if self.map_xlim is None:
      self.map_xlim = self.map_ax.get_xlim()

    if self.map_ylim is None:  
      self.map_ylim = self.map_ax.get_ylim()
      
    self.map_ax.set_xlim(self.map_xlim)
    self.map_ax.set_ylim(self.map_ylim)
      
  def _update_map_ax(self):
    # Updates the map axis for changes in *xidx* or *tidx*. This 
    # involves changing the title to display the current date
    if self.map_title is None:
      time_label = self.time_labels[self.tidx]
      self.map_ax.set_title('time : %s' % time_label,
                            fontsize=self.fontsize)
    else:
      self.map_ax.set_title(self.map_title,
                            fontsize=self.fontsize)

  def _init_image(self):
    # Initially plots the vertical deformation image.
    #
    # CALL THIS AFTER *_init_map_ax*
    #
    self.x_itp = [np.linspace(self.map_xlim[0],
                              self.map_xlim[1],
                              self.image_resolution),
                  np.linspace(self.map_ylim[0],
                              self.map_ylim[1],
                              self.image_resolution)]
    data_itp = _grid_interp_data(self.data_sets[0][self.tidx,:,2],
                                 self.x,self.x_itp[0],self.x_itp[1])
    if self.image_clim is None:
      # if vmin and vmax are None then the color bounds will be 
      # updated each time the artists are redrawn
      image_clim = data_itp.min(),data_itp.max()
    else:  
      image_clim = self.image_clim

    self.image = self.map_ax.imshow(
                   data_itp,
                   extent=(self.map_xlim+self.map_ylim),
                   interpolation='bicubic',
                   origin='lower',
                   vmin=image_clim[0],vmax=image_clim[1],
                   cmap=self.image_cmap,
                   zorder=1)
    # Allocate a space in the figure for the colorbar if a colorbar 
    # has not already been generated.
    if not hasattr(self,'cbar'):
      self.cbar = self.map_fig.colorbar(self.image,ax=self.map_ax)  
    else:
      self.cbar = self.map_fig.colorbar(self.image,cax=self.cbar.ax)  
      
    if self.units is None:
      image_clabel = 'vertical'
    else:
      image_clabel = 'vertical [%s]' % self.units
      
    self.cbar.set_clim(image_clim)
    self.cbar.set_label(image_clabel,fontsize=self.fontsize)
    self.cbar.ax.tick_params(labelsize=self.fontsize)
    self.cbar.solids.set_rasterized(True)

  def _update_image(self):
    # Update the vertical deformation image for changes in *tidx* or 
    # *xidx*. This changes the data for the image and updates the 
    # colorbar
    data_itp = _grid_interp_data(self.data_sets[0][self.tidx,:,2],
                                 self.x,self.x_itp[0],self.x_itp[1])
    self.image.set_data(data_itp)
    if self.image_clim is None:
      # *image_clim* are the user specified color bounds. if they are 
      # None then the color bounds will be updated each time the 
      # artists are redrawn
      image_clim = data_itp.min(),data_itp.max()
    else:  
      image_clim = self.image_clim

    self.image.set_clim(image_clim)
    self.cbar.set_clim(image_clim)
    
  def _init_scatter(self):
    # Plots the scatter points at the base of each vector showing the 
    # vertical deformation for the second data set. If there is only 
    # one data set then this function does nothing.
    # 
    # CALL THIS AFTER *_init_image*
    #
    if len(self.data_sets) < 2:
      self.scatter = None 
      return

    sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cbar.get_cmap())
    # use scatter points to show z for second data set 
    colors = sm.to_rgba(self.data_sets[1][self.tidx,:,2])
    self.scatter = self.map_ax.scatter(self.x[:,0],self.x[:,1],
                                       c=colors,s=self.scatter_size,
                                       zorder=2,edgecolor=self.colors[1])

  def _update_scatter(self):
    # Updates the scatter points for changes in *tidx* or *xidx*. This 
    # just changes the face color
    # 
    # CALL THIS AFTER *_update_image*
    #
    if len(self.data_sets) < 2:
      return

    sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cbar.get_cmap())
    colors = sm.to_rgba(self.data_sets[1][self.tidx,:,2])
    self.scatter.set_facecolors(colors)

  def _init_quiver(self):
    # Initially plots the horizontal deformation vectors and a key
    self.quiver = []
    for si in range(len(self.data_sets)):
      q = Quiver(self.map_ax,self.x[:,0],self.x[:,1],
                 self.data_sets[si][self.tidx,:,0],
                 self.data_sets[si][self.tidx,:,1],
                 scale=self.quiver_scale,  
                 width=0.005,
                 sigma=(self.sigma_sets[si][self.tidx,:,0],
                        self.sigma_sets[si][self.tidx,:,1],
                        np.zeros(self.x.shape[0])), 
                 color=self.colors[si],
                 ellipse_kwargs={'edgecolors':'k','zorder':2+si},
                 zorder=3+si)
      self.map_ax.add_collection(q,autolim=True)
      self.map_ax.autoscale_view()                 
      self.quiver += [q]                        
      if si == 0:
        # plot quiver key for the first data set
        if self.units is None:
          quiver_key_label = '%s' % self.quiver_key_length
        else:
          quiver_key_label = '%s %s' % (self.quiver_key_length,
                                        self.units)
          
        self.key = self.map_ax.quiverkey(
                     self.quiver[si],
                     self.quiver_key_pos[0],
                     self.quiver_key_pos[1],
                     self.quiver_key_length,
                     quiver_key_label,zorder=3,labelsep=0.05,
                     fontproperties={'size':self.fontsize})
                     
  def _update_quiver(self):
    # Updates the deformation vectors for changes in *tidx* or *xidx* 
    for si in range(len(self.data_sets)):
      self.quiver[si].set_UVC(
                        self.data_sets[si][self.tidx,:,0],
                        self.data_sets[si][self.tidx,:,1],
                        sigma=(self.sigma_sets[si][self.tidx,:,0],
                               self.sigma_sets[si][self.tidx,:,1],
                               np.zeros(self.x.shape[0])))

  def _init_pickers(self):
    # Initially plots the picker artists, which are used to select
    # station by clicking on them. The currently picked station has a
    # slightly larger marker. Labels are also created for each picker
    self.pickers = []
    self.text = []
    for i,(si,xi) in enumerate(zip(self.station_labels,self.x)):
      if self.highlight:
        # make pickers and text visible
        if i == self.xidx:
          # make the picker larger if i==xidx
          self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                           picker=10,
                                           markersize=15,
                                           markerfacecolor='k',
                                           markeredgecolor='k')

        else:          
          self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                           picker=10,
                                           markersize=5,
                                           markerfacecolor='k',
                                           markeredgecolor='k')
                                           
        self.text += [self.map_ax.text(xi[0],xi[1],si,
                                       fontsize=self.fontsize,
                                       color=(0.4,0.4,0.4))]
      else:                                       
        # just create the pickers and dont make them visible
        self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                         picker=10,
                                         markersize=0,
                                         markerfacecolor='k',
                                         markeredgecolor='k')

  def _update_pickers(self):
    # Change the larger picker according to the current xidx
    if self.highlight:
      # only make the changes if highlight is True
      for i,p in enumerate(self.pickers):
        if i == self.xidx:
          p.set_markersize(15)

        else:
          p.set_markersize(5)  
        
  def _init_lines(self):
    # Initially plots the time series for each component of 
    # deformation
    self.line1,self.line2,self.line3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.line1 += self.ts_ax[0].plot(
                      self.t,self.data_sets[si][:,self.xidx,0],
                      color=self.colors[si],
                      label=self.dataset_labels[si],
                      linestyle=self.line_styles[si],
                      marker=self.line_markers[si],
                      linewidth=1.0,
                      zorder=3)
      self.line2 += self.ts_ax[1].plot(
                      self.t,self.data_sets[si][:,self.xidx,1],
                      color=self.colors[si],
                      label=self.dataset_labels[si],
                      linestyle=self.line_styles[si],
                      marker=self.line_markers[si],
                      linewidth=1.0,
                      zorder=3)
      self.line3 += self.ts_ax[2].plot(
                      self.t,self.data_sets[si][:,self.xidx,2],
                      color=self.colors[si],
                      label=self.dataset_labels[si],
                      linestyle=self.line_styles[si],
                      marker=self.line_markers[si],
                      linewidth=1.0,
                      zorder=3)
    
  def _update_lines(self):
    # Updates the deformation time series for changes in *tidx* or 
    # *xidx*.
    for si in range(len(self.data_sets)):
      self.line1[si].set_data(self.t,self.data_sets[si][:,self.xidx,0])
      # relabel in case the data_set order has switched
      self.line1[si].set_label(self.dataset_labels[si])                     
      self.line2[si].set_data(self.t,self.data_sets[si][:,self.xidx,1])
      self.line2[si].set_label(self.dataset_labels[si])                     
      self.line3[si].set_data(self.t,self.data_sets[si][:,self.xidx,2])
      self.line3[si].set_label(self.dataset_labels[si])                     
  
  def _init_err(self):
    # Initially plots the confidence interval for each deformation 
    # component.
    self.err1,self.err2,self.err3 = [],[],[]
    for si in range(len(self.data_sets)):
      if self.error_styles[si] == 'fill':
        self.err1 += [self.ts_ax[0].fill_between(
                       self.t,
                       self.data_sets[si][:,self.xidx,0] -
                       self.sigma_sets[si][:,self.xidx,0],
                       self.data_sets[si][:,self.xidx,0] +
                       self.sigma_sets[si][:,self.xidx,0],
                       edgecolor='none',
                       facecolor=self.colors[si],alpha=0.2,
                       zorder=2)]
        self.err2 += [self.ts_ax[1].fill_between(
                       self.t,
                       self.data_sets[si][:,self.xidx,1] -
                       self.sigma_sets[si][:,self.xidx,1],
                       self.data_sets[si][:,self.xidx,1] +
                       self.sigma_sets[si][:,self.xidx,1],
                       edgecolor='none',
                       facecolor=self.colors[si],alpha=0.2,
                       zorder=2)]
        self.err3 += [self.ts_ax[2].fill_between(
                       self.t,
                       self.data_sets[si][:,self.xidx,2] -
                       self.sigma_sets[si][:,self.xidx,2],
                       self.data_sets[si][:,self.xidx,2] +
                       self.sigma_sets[si][:,self.xidx,2],
                       edgecolor='none',
                       facecolor=self.colors[si],alpha=0.2,
                       zorder=2)]

      elif self.error_styles[si] == 'bar':
        self.err1 += [self.ts_ax[0].errorbar(
                       self.t,
                       self.data_sets[si][:,self.xidx,0],
                       self.sigma_sets[si][:,self.xidx,0],
                       linestyle='None',
                       color=self.colors[si],
                       zorder=2)]
        self.err2 += [self.ts_ax[1].errorbar(
                       self.t,
                       self.data_sets[si][:,self.xidx,1],
                       self.sigma_sets[si][:,self.xidx,1],
                       linestyle='None',
                       color=self.colors[si],
                       zorder=2)]
        self.err3 += [self.ts_ax[2].errorbar(
                       self.t,
                       self.data_sets[si][:,self.xidx,2],
                       self.sigma_sets[si][:,self.xidx,2],
                       linestyle='None',
                       color=self.colors[si],
                       zorder=2)]

      elif self.error_styles[si] == 'None':
        return
      
      else:
        raise ValueError(
          'elements of *error_styles* must be "fill", "bar", or "None"')

  def _update_err(self):
    # Updates the confidence intervals for changes in *xidx* or 
    # *tidx*. Unfortunately, the only way to update these artists is 
    # to remove and replot them.
    for f in self.err1: f.remove()
    for f in self.err2: f.remove()
    for f in self.err3: f.remove()
    self._init_err()
    
  def _remove_artists(self):
    # This function should remove EVERY artist
    for f in self.err1: f.remove()
    for f in self.err2: f.remove()
    for f in self.err3: f.remove()
    for l in self.line1: l.remove()
    for l in self.line2: l.remove()
    for l in self.line3: l.remove()
    for q in self.quiver: q.remove()
    for p in self.pickers: p.remove()
    for t in self.text: t.remove()
    self.key.remove()
    self.image.remove()
    if self.scatter is not None:
      self.scatter.remove()        

    self.cbar.ax.clear()

  def _init(self):
    # Calls every _init function
    self._init_pickers()
    self._init_map_ax()
    self._init_quiver()
    self._init_lines()
    self._init_err()
    self._init_image()
    self._init_scatter()
    self._init_ts_ax()
    self.map_fig.canvas.draw()
    self.ts_fig.tight_layout()
    self.ts_fig.canvas.draw()

  def update(self):
    # Calls every _update function
    self._update_pickers()
    self._update_map_ax()
    self._update_quiver()
    self._update_lines()
    self._update_err()
    self._update_image()
    self._update_scatter()
    self._update_ts_ax()
    self.ts_fig.canvas.draw()
    self.map_fig.canvas.draw()

  def hard_update(self):
    # Removes all artists and replots them. This is slower but it 
    # properly updates the figures for any changes to the configurable 
    # parameters.
    self._remove_artists()
    self._init()
    
  @without_interactivity
  def configure(self):
    Configurable.configure(self)
  
  def save_frames(self,dir):
    # saves each frame as a jpeg image in the direction *dir*
    Nt = self.data_sets[0].shape[0]
    for i in range(Nt):
      self.tidx = i
      self.update()
      fname = '%06d.jpeg' % i
      logger.info('saving file %s/%s' % (dir,fname))
      plt.savefig(dir+'/'+fname)

  def on_pick(self,event):
    # This function is called when the mouse is clicked
    for i,v in enumerate(self.pickers):
      if event.artist == v:
        self.xidx = i
        break

    self.update()    

  def on_key_press(self,event):
    # This function is called when a key is pressed
    if event.key == 'right':
      self.tidx += 1
      self.update()

    elif event.key == 'ctrl+right':
      self.tidx += 10
      self.update()

    elif event.key == 'alt+right':
      self.tidx += 100
      self.update()

    elif event.key == 'left':
      self.tidx -= 1
      self.update()

    elif event.key == 'ctrl+left':
      self.tidx -= 10
      self.update()

    elif event.key == 'alt+left':
      self.tidx -= 100
      self.update()

    elif event.key == 'up':
      self.xidx += 1
      self.update()

    elif event.key == 'ctrl+up':
      self.xidx += 10
      self.update()

    elif event.key == 'alt+up':
      self.xidx += 100
      self.update()

    elif event.key == 'down':
      self.xidx -= 1
      self.update()

    elif event.key == 'ctrl+down':
      self.xidx -= 10
      self.update()

    elif event.key == 'alt+down':
      self.xidx -= 100
      self.update()

    elif event.key == 'h':
      # toggle station highlighter
      self.highlight = not self.highlight
      self.hard_update()

    elif event.key == 'c':
      # cycle data arrays 
      self.data_sets = _roll(self.data_sets)
      self.dataset_labels = _roll(self.dataset_labels)
      self.sigma_sets = _roll(self.sigma_sets)
      self.hard_update()
      
    elif event.key == 'v':
      # toggle vertical deformation 
      if self.image_cmap is _blank_cmap:
        self.image_cmap = self._previous_cmap
      
      else:
        self._previous_cmap = self.image_cmap
        self.image_cmap = _blank_cmap  
        
      self.hard_update()

    elif event.key == 'r':
      # refresh  
      self.hard_update()
      
    elif event.key == 'enter':
      self.configure()
      self.hard_update()
      
    else:
      # do nothing
      return


def interactive_vector_viewer(*args,**kwargs):
  ''' 
  wrapper to initiate and show an InteractiveViewer
  '''
  iv = InteractiveVectorViewer(*args,**kwargs)
  iv.connect()
  return   

