''' 
This module provides functions for plotting strain
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pygeons.plot.ivector import disable_default_key_bindings
from pygeons.plot.ivector import without_interactivity
from pygeons.plot.ivector import one_sigfig
from pygeons.plot.ivector import Configurable
from pygeons.plot.strain_glyph import strain_glyph
from scipy.spatial import cKDTree
import logging
logger = logging.getLogger(__name__)


class InteractiveStrainViewer(Configurable):
  ''' 
----------- PyGeoNS Interactive Strain Viewer (PISV) ----------------

An interactive figure for viewing the spatial and temporal patterns in
strain.

Controls :
  Left : move back 1 time step (Ctrl-Left and Alt-Left move back 10 and 
    100 respectively)
  
  Right : move forward 1 time step (Ctrl-Right and Alt-Right move 
    forward 10 and 100 respectively)
  
  Up : Move forward 1 station (Ctrl-Left and Alt-Left move back 10 and 
    100 respectively)
            
  Down : Move back 1 station (Ctrl-Right and Alt-Right move forward 10 
    and 100 respectively)
  
  R : redraw figures
  
  H : Hide station marker
  
  Enter : Edit parameters through the command line
  
Notes :
  Stations may also be selected by clicking on them.
  
  Exit PISV by closing the figures.
    
  Key bindings only work when the active window is one of the PISV 
  figures.
  
---------------------------------------------------------------------
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
               exx,eyy,exy,
               sxx=None,syy=None,sxy=None,
               scale=None,
               units=None,
               time_labels=None,
               station_labels=None,
               fontsize=10,
               map_ax=None,
               ts_ax=None,
               ts_title=None,
               map_title=None,
               map_ylim=None,
               map_xlim=None,
               compression_color='r',
               extension_color='b',
               alpha=0.2,
               vertices=100,
               key_magnitude=None,
               key_position=(0.15,0.2),
               snr_mask=True):
    ''' 
    interactively views strain which is time and space dependent
    
    Parameters
    ----------
      t : (Nt,) array
      x : (Nx,2) array
      exx,eyy,exy : (Nt,Nx) array
      sxx,syy,sxy : (Nt,Nx) array
      **kwargs        
    '''
    ## SET T AND X
    #################################################################
    self.t = np.asarray(t)
    self.x = np.asarray(x)
    Nx,Nt = len(x),len(t)
    
    ## SET DATA_SET AND SIGMA_SET
    #################################################################
    if sxx is None: sxx = np.zeros((Nt,Nx))
    if syy is None: syy = np.zeros((Nt,Nx))
    if sxy is None: sxy = np.zeros((Nt,Nx))

    tpl = (exx[:,:,None],eyy[:,:,None],exy[:,:,None])
    self.data_set = np.concatenate(tpl,axis=2)
    tpl = (sxx[:,:,None],syy[:,:,None],sxy[:,:,None])
    self.sigma_set = np.concatenate(tpl,axis=2)

    ## SET MAP_AX, TS_AX, MAP_FIG, AND TS_FIG
    #################################################################
    if map_ax is None:
      # gives a white background 
      map_fig,map_ax = plt.subplots(num='Map View',facecolor='white')
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
    # station labels
    if station_labels is None:
      station_labels = ['%04d' % i for i in range(Nx)]
    
    self.station_labels = station_labels  
              
    # time labels  
    if time_labels is None:
      time_labels = [str(i) for i in self.t]

    self.time_labels = time_labels

    # scale
    if scale is None:
      # Get an idea of what the typical strain magnitudes are
      mags = np.linalg.norm(self.data_set,axis=2)
      mag = max(np.nanmean(mags),1e-10)
      # find the average distance between points
      if Nx <= 1:
        dist = 1.0
      else:
        kd = cKDTree(self.x) 
        dist = np.mean(kd.query(self.x,2)[0][:,1])
    
      scale = dist/mag
      
    self.scale = scale
    
    # key magnitude
    if key_magnitude is None:  
      mags = np.linalg.norm(self.data_set,axis=2)
      mag = max(np.nanmean(mags),1e-10)
      key_magnitude = one_sigfig(mag)
      
    self.key_magnitude = key_magnitude
    
    # set additional properties
    self.highlight = True 
    self.tidx = 0
    self.xidx = 0
    self.scale = scale
    self.map_title = map_title
    self.map_xlim = map_xlim
    self.map_ylim = map_ylim
    self.ts_title = ts_title
    self.fontsize = fontsize
    self.units = units
    self.compression_color = compression_color
    self.extension_color = extension_color
    self.alpha = alpha
    self.vertices = vertices
    self.key_position = key_position
    self.snr_mask = snr_mask

    # initiate all artists
    self._init()
    # turn off MPL key bindings and use my own
    disable_default_key_bindings()
    # display help
    logger.info(self.__doc__)

  def connect(self):
    self.ts_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.ts_fig.canvas.mpl_connect('pick_event',self.on_pick)
    self.map_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.map_fig.canvas.mpl_connect('pick_event',self.on_pick)

  def _init_ts_ax(self):
    # Initially configures the time series axes. This involves setting
    # the titles, labels, and scaling to fir the displayed data
    #
    # CALL THIS AFTER *_init_lines*
    #
    if self.units is None:
      ts_ylabel_0 = 'east normal'
      ts_ylabel_1 = 'north normal'
      ts_ylabel_2 = 'east-north shear'

    else:
      ts_ylabel_0 = 'east normal\n[%s]' % self.units
      ts_ylabel_1 = 'north normal\n[%s]' % self.units
      ts_ylabel_2 = 'east-north shear\n[%s]' % self.units

    self.ts_ax[2].set_xlabel('time')
    self.ts_ax[0].set_ylabel(ts_ylabel_0)
    self.ts_ax[1].set_ylabel(ts_ylabel_1)
    self.ts_ax[2].set_ylabel(ts_ylabel_2)
    self.ts_ax[0].grid(c='0.5',alpha=0.5)
    self.ts_ax[1].grid(c='0.5',alpha=0.5)
    self.ts_ax[2].grid(c='0.5',alpha=0.5)
    self.ts_ax[0].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[1].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[2].xaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[0].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[1].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[2].yaxis.label.set_fontsize(self.fontsize)
    self.ts_ax[2].title.set_fontsize(self.fontsize)
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

    # hide xtick labels for the top two axes
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
    # Update the time series axes for changes in *tidx* or *xidx*.
    # THis involves changes the axes titles and rescaling for the new
    # data being displayed
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
    # Initially configure the map view axis.  THis involves setting
    # the titles, labels, and scaling to fit the plotted data.
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
    # involves changing the title to display the current time
    if self.map_title is None:
      time_label = self.time_labels[self.tidx]
      self.map_ax.set_title('time : %s' % time_label,
                            fontsize=self.fontsize)
    else:
      self.map_ax.set_title(self.map_title,
                            fontsize=self.fontsize)

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

  def _init_key(self):
    # Initially create the strain glyph key
    self.glyph_key = []
    mag = self.key_magnitude
    units = self.units
    strain = [mag,-mag,0.0]
    sigma = [0.25*mag,0.25*mag,0.25*mag]

    key_pos_axes = self.key_position
    key_pos_display = self.map_ax.transAxes.transform(key_pos_axes)
    key_pos_data = self.map_ax.transData.inverted().transform(key_pos_display)
    posx,posy = key_pos_data
    self.glyph_key += strain_glyph(key_pos_data,strain,sigma=sigma,
                                   scale=self.scale,
                                   ext_color=self.extension_color,
                                   cmp_color=self.compression_color,
                                   alpha=self.alpha,
                                   vert=self.vertices)
    if units is None:
      text_str = '%s' % mag
    else:
      text_str = '%s %s' % (mag,units)
      
    textx = posx + 1.1*mag*self.scale
    texty = posy
    self.glyph_key += [Text(textx,texty,text_str,
                            fontsize=10,
                            color=self.extension_color)]
    textx = posx
    texty = posy + 1.1*mag*self.scale
    self.glyph_key += [Text(textx,texty,'-' + text_str,
                            fontsize=10,
                            color=self.compression_color)]

    for a in self.glyph_key: self.map_ax.add_artist(a)
  
  def _init_strain(self): 
    # Initially plots the strain glyphs
    self.glyphs = []
    strain = self.data_set[self.tidx,:,:]
    sigma = self.sigma_set[self.tidx,:,:]
    for args in zip(self.x,strain,sigma):
      self.glyphs += strain_glyph(*args,
                                  scale=self.scale,
                                  ext_color=self.extension_color,
                                  cmp_color=self.compression_color,
                                  alpha=self.alpha,
                                  vert=self.vertices,
                                  snr_mask=self.snr_mask)
       
    for a in self.glyphs: self.map_ax.add_artist(a)

  def _update_strain(self):
    # Remove any existing strain artists and replot
    for a in self.glyphs: a.remove()
    self._init_strain()

  def _init_lines(self):
    # Create strain time series lines
    self.line1, = self.ts_ax[0].plot(
                    self.t,
                    self.data_set[:,self.xidx,0],
                    color='k',
                    linestyle='-',
                    linewidth=1.0)
    self.line2, = self.ts_ax[1].plot(
                    self.t,
                    self.data_set[:,self.xidx,1],
                    color='k',
                    linestyle='-',
                    linewidth=1.0)
    self.line3, = self.ts_ax[2].plot(
                    self.t,
                    self.data_set[:,self.xidx,2],
                    color='k',
                    linestyle='-',
                    linewidth=1.0)

  def _update_lines(self):
    # Update strain time series lines
    self.line1.set_data(self.t,
                        self.data_set[:,self.xidx,0])
    # relabel in case the data_set order has switched
    self.line2.set_data(self.t,
                        self.data_set[:,self.xidx,1])
    self.line3.set_data(self.t,
                        self.data_set[:,self.xidx,2])

  def _init_fill(self):
    # Create uncertainties in the strain timeseries
    self.fill1 = self.ts_ax[0].fill_between(
                   self.t,
                   self.data_set[:,self.xidx,0] -
                   self.sigma_set[:,self.xidx,0],
                   self.data_set[:,self.xidx,0] +
                   self.sigma_set[:,self.xidx,0],
                   edgecolor='none',
                   color='k',alpha=0.2)
    self.fill2 = self.ts_ax[1].fill_between(
                   self.t,
                   self.data_set[:,self.xidx,1] -
                   self.sigma_set[:,self.xidx,1],
                   self.data_set[:,self.xidx,1] +
                   self.sigma_set[:,self.xidx,1],
                   edgecolor='none',
                   color='k',alpha=0.2)
    self.fill3 = self.ts_ax[2].fill_between(
                   self.t,
                   self.data_set[:,self.xidx,2] -
                   self.sigma_set[:,self.xidx,2],
                   self.data_set[:,self.xidx,2] +
                   self.sigma_set[:,self.xidx,2],
                   edgecolor='none',
                   color='k',alpha=0.2)

  def _update_fill(self):
    # Replot uncertainties in the strain timeseries
    self.fill1.remove()
    self.fill2.remove()
    self.fill3.remove()
    self._init_fill()

  def _init(self):
    self._init_pickers()
    self._init_key()
    self._init_strain()
    self._init_lines()
    self._init_fill()
    self._init_map_ax()
    self._init_ts_ax()
    self.map_fig.canvas.draw()
    self.map_ax.autoscale_view()
    self.ts_fig.tight_layout()
    self.ts_fig.canvas.draw()

  def update(self):
    self._update_pickers()
    self._update_strain()
    self._update_lines()
    self._update_fill()
    self._update_map_ax()
    self._update_ts_ax()
    self.ts_fig.canvas.draw()
    self.map_fig.canvas.draw()

  def _remove_artists(self):
    self.line1.remove()
    self.line2.remove()
    self.line3.remove()
    self.fill1.remove()
    self.fill2.remove()
    self.fill3.remove()
    for a in self.pickers: a.remove()
    for a in self.text: a.remove()
    for a in self.glyphs: a.remove()
    for a in self.glyph_key: a.remove()

  def hard_update(self):
    self._remove_artists() 
    self._init()

  @without_interactivity
  def configure(self):
    Configurable.configure(self)

  def save_frames(self,dir):
    # saves each frame as a jpeg image in the direction *dir*
    Nt = self.data_set.shape[0]
    for i in range(Nt):
      self.tidx = i
      self.update()
      fname = '%06d.jpeg' % i
      logger.info('saving file %s/%s' % (dir,fname))
      plt.savefig(dir+'/'+fname)

  def on_pick(self,event):
    for i,v in enumerate(self.pickers):
      if event.artist == v:
        self.xidx = i
        break

    self.update()

  def on_key_press(self,event):
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
      self.highlight = not self.highlight
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

def interactive_strain_viewer(*args,**kwargs):
  ''' 
  wrapper to initiate and show an InteractiveStrainViewer
  '''
  iv = InteractiveStrainViewer(*args,**kwargs)
  iv.connect()
  plt.show()
  return

