''' 
This module provides functions for plotting strain
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pygeons.plot.iview import disable_default_key_bindings
from pygeons.plot.iview import without_interactivity
from pygeons.plot.iview import one_sigfig
from pygeons.plot.rin import restricted_input
from pygeons.plot.strain_glyph import strain_glyph
from scipy.spatial import cKDTree
import os


class InteractiveStrainViewer(object):
  ''' 
              ----------------------------------------
              PyGeoNS Interactive Strain Viewer (PISV)
              ----------------------------------------

An interactive figure for viewing the spatial and temporal patterns in 
strain.

Controls
--------
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

Enter : Disables figures and allows configurable parameters to be 
  edited through the command line. Variables can be defined using any 
  functions in the numpy, matplotlib, or base python namespace

Notes
-----
Stations may also be selected by clicking on them.

Exit PISV by closing the figures.
  
Key bindings only work when the active window is one of the PISV 
figures.

---------------------------------------------------------------------
  '''
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
               key_position=(0.1,0.1)):
    ''' 
    interactively views strain which is time and space dependent
    
    Parameters
    ----------
      t : (Nt,) array
        Observation times

      x : (Nx,2) array
        Observation positions
        
      exx,eyy,exy : (Nt,Nx) array
        Strain tensor components

      sxx,syy,sxy : (Nt,Nx) array
        One standard deviation uncertainty on the strain components
        
      scale : float
        Increases the size of the strain markers
         
      map_ax : Axis instance
        Axis where map view will be plotted

      map_title : str
        Replaces the default title for the map view plot

      map_ylim : (2,) array
        Y limits for the map view plot
      
      xlim : (2,) array
        X limits for the map view plot
      
      fontsize : float
        Controls all fontsizes
      
      key_magnitude : float
        strain magnitude for the key
        
      key_position : tuple    
        position of the key in axis coordinates
        
    '''
    # time and space arrays
    self.t = np.asarray(t)
    self.x = np.asarray(x)
    Nx,Nt = len(x),len(t)
    
    if sxx is None: sxx = np.zeros((Nt,Nx))
    if syy is None: syy = np.zeros((Nt,Nx))
    if sxy is None: sxy = np.zeros((Nt,Nx))

    tpl = (exx[:,:,None],eyy[:,:,None],exy[:,:,None])
    self.data_set = np.concatenate(tpl,axis=2)
    tpl = (sxx[:,:,None],syy[:,:,None],sxy[:,:,None])
    self.sigma_set = np.concatenate(tpl,axis=2)

    # map view axis and figure
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

    # station names used for the time series plots
    if station_labels is None:
      station_labels = np.arange(len(self.x)).astype(str)
              
    if time_labels is None:
      time_labels = np.array(self.t).astype(str)

    self.time_labels = list(time_labels)
    self.station_labels = list(station_labels)

    # estimate a good scale for the strain glyphs
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
      
    if key_magnitude is None:  
      mags = np.linalg.norm(self.data_set,axis=2)
      mag = max(np.nanmean(mags),1e-10)
      key_magnitude = one_sigfig(mag)
      
    # this dictionary contains plot configuration parameters which may 
    # be interactively changed
    self.config = {}
    self.config['highlight'] = True 
    self.config['tidx'] = 0
    self.config['xidx'] = 0
    self.config['scale'] = scale
    self.config['map_title'] = map_title
    self.config['map_xlim'] = map_xlim
    self.config['map_ylim'] = map_ylim
    self.config['ts_title'] = ts_title
    self.config['fontsize'] = fontsize
    self.config['units'] = units
    self.config['compression_color'] = compression_color
    self.config['extension_color'] = extension_color
    self.config['alpha'] = alpha
    self.config['vertices'] = vertices
    self.config['key_magnitude'] = key_magnitude
    self.config['key_position'] = key_position
    self._init()
    disable_default_key_bindings()
    print(self.__doc__)

  def connect(self):
    self.ts_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.ts_fig.canvas.mpl_connect('pick_event',self.on_pick)
    self.map_fig.canvas.mpl_connect('key_press_event',self.on_key_press)
    self.map_fig.canvas.mpl_connect('pick_event',self.on_pick)

  def save_frames(self,dir):
    ''' 
    saves each frame as a jpeg image in the direction *dir*
    '''
    Nt = self.data_set.shape[0]
    for i in range(Nt):
      self.config['tidx'] = i
      self.update()
      fname = '%06d.jpeg' % i
      print('saving file %s/%s' % (dir,fname))
      plt.savefig(dir+'/'+fname)
    
    print('done')  

  def _init_ts_ax(self):
    # call after _init_lines
    if self.config['units'] is None:
      ts_ylabel_0 = 'east normal'
      ts_ylabel_1 = 'north normal'
      ts_ylabel_2 = 'east-north shear'

    else:
      ts_ylabel_0 = 'east normal [%s]' % self.config['units']
      ts_ylabel_1 = 'north normal [%s]' % self.config['units']
      ts_ylabel_2 = 'east-north shear [%s]' % self.config['units']

    self.ts_ax[2].set_xlabel('time')
    self.ts_ax[0].set_ylabel(ts_ylabel_0)
    self.ts_ax[1].set_ylabel(ts_ylabel_1)
    self.ts_ax[2].set_ylabel(ts_ylabel_2)
    self.ts_ax[0].grid()
    self.ts_ax[1].grid()
    self.ts_ax[2].grid()
    self.ts_ax[0].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[1].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[2].xaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[0].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[1].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[2].yaxis.label.set_fontsize(self.config['fontsize'])
    self.ts_ax[2].title.set_fontsize(self.config['fontsize'])
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
    if self.config['map_title'] is None:
      time_label = self.time_labels[self.config['tidx']]
      self.map_ax.set_title('time : %s' % time_label,
                        fontsize=self.config['fontsize'])
    else:
      self.map_ax.set_title(self.config['map_title'],
                        fontsize=self.config['fontsize'])

  def _init_marker(self):
    self.marker, = self.map_ax.plot(self.x[self.config['xidx'],0],
                                    self.x[self.config['xidx'],1],'ko',
                                    markersize=20*self.config['highlight'])

  def _update_marker(self):
    # updates for:
    #   xidx
    #   highlight
    self.marker.set_data(self.x[self.config['xidx'],0],
                         self.x[self.config['xidx'],1])
    self.marker.set_markersize(20*self.config['highlight'])

  def _init_pickers(self):
    # pickable artists
    self.pickers = []
    for xi in self.x:
      self.pickers += self.map_ax.plot(xi[0],xi[1],'.',
                                       picker=10,
                                       markersize=0)

  def _init_key(self):
    self.glyph_key = []
    mag = self.config['key_magnitude']
    units = self.config['units']
    strain = [mag,-mag,0.0]
    sigma = [0.25*mag,0.25*mag,0.25*mag]

    key_pos_axes = self.config['key_position']
    key_pos_display = self.map_ax.transAxes.transform(key_pos_axes)
    key_pos_data = self.map_ax.transData.inverted().transform(key_pos_display)
    posx,posy = key_pos_data
    self.glyph_key += strain_glyph(key_pos_data,strain,sigma=sigma,
                                   scale=self.config['scale'],
                                   ext_color=self.config['extension_color'],
                                   cmp_color=self.config['compression_color'],
                                   alpha=self.config['alpha'],
                                   vert=self.config['vertices'])
    if units is None:
      text_str = '%s' % mag
    else:
      text_str = '%s %s' % (mag,units)
      
    textx = posx + 1.1*mag*self.config['scale']
    texty = posy
    self.glyph_key += [Text(textx,texty,text_str,
                            fontsize=10,
                            color=self.config['extension_color'])]
    textx = posx
    texty = posy + 1.1*mag*self.config['scale']
    self.glyph_key += [Text(textx,texty,'-' + text_str,
                            fontsize=10,
                            color=self.config['compression_color'])]

    for a in self.glyph_key: self.map_ax.add_artist(a)
  
  def _update_key(self):
    for a in self.glyph_key: a.remove()
    self._init_key()

  def _init_strain(self): 
    self.glyphs = []
    strain = self.data_set[self.config['tidx'],:,:]
    sigma = self.sigma_set[self.config['tidx'],:,:]
    for args in zip(self.x,strain,sigma):
      self.glyphs += strain_glyph(*args,
                                  scale=self.config['scale'],
                                  ext_color=self.config['extension_color'],
                                  cmp_color=self.config['compression_color'],
                                  alpha=self.config['alpha'],
                                  vert=self.config['vertices'])
       
    for a in self.glyphs: self.map_ax.add_artist(a)

  def _update_strain(self):
    # remove any existing strain artists and replot
    for a in self.glyphs: a.remove()
    self._init_strain()

  def _init_lines(self):
    self.line1, = self.ts_ax[0].plot(
                    self.t,
                    self.data_set[:,self.config['xidx'],0],
                    color='k',
                    ls='-')
    self.line2, = self.ts_ax[1].plot(
                    self.t,
                    self.data_set[:,self.config['xidx'],1],
                    color='k',
                    ls='-')
    self.line3, = self.ts_ax[2].plot(
                    self.t,
                    self.data_set[:,self.config['xidx'],2],
                    color='k',
                    ls='-')

  def _update_lines(self):
    # updates for:
    #   xidx
    self.line1.set_data(self.t,
                        self.data_set[:,self.config['xidx'],0])
    # relabel in case the data_set order has switched
    self.line2.set_data(self.t,
                        self.data_set[:,self.config['xidx'],1])
    self.line3.set_data(self.t,
                        self.data_set[:,self.config['xidx'],2])

  def _init_fill(self):
    self.fill1 = self.ts_ax[0].fill_between(
                   self.t,
                   self.data_set[:,self.config['xidx'],0] -
                   self.sigma_set[:,self.config['xidx'],0],
                   self.data_set[:,self.config['xidx'],0] +
                   self.sigma_set[:,self.config['xidx'],0],
                   edgecolor='none',
                   color='k',alpha=0.5)
    self.fill2 = self.ts_ax[1].fill_between(
                   self.t,
                   self.data_set[:,self.config['xidx'],1] -
                   self.sigma_set[:,self.config['xidx'],1],
                   self.data_set[:,self.config['xidx'],1] +
                   self.sigma_set[:,self.config['xidx'],1],
                   edgecolor='none',
                   color='k',alpha=0.5)
    self.fill3 = self.ts_ax[2].fill_between(
                   self.t,
                   self.data_set[:,self.config['xidx'],2] -
                   self.sigma_set[:,self.config['xidx'],2],
                   self.data_set[:,self.config['xidx'],2] +
                   self.sigma_set[:,self.config['xidx'],2],
                   edgecolor='none',
                   color='k',alpha=0.5)

  def _update_fill(self):
    # updates for:
    #   xidx
    self.fill1.remove()
    self.fill2.remove()
    self.fill3.remove()
    self._init_fill()

  def _init(self):
    self._init_pickers()
    self._init_marker()
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
    self._update_marker()
    self._update_key()
    self._update_strain()
    self._update_lines()
    self._update_fill()
    self._update_map_ax()
    self._update_ts_ax()
    self.ts_fig.canvas.draw()
    self.map_fig.canvas.draw()

  def _remove_artists(self):
    self.marker.remove()
    self.line1.remove()
    self.line2.remove()
    self.line3.remove()
    self.fill1.remove()
    self.fill2.remove()
    self.fill3.remove()
    for a in self.pickers: a.remove()
    for a in self.glyphs: a.remove()
    for a in self.glyph_key: a.remove()

  def hard_update(self):
    self._remove_artists() 
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
      print('the following error was raised when evaluating the above expression:\n    %s\n' % repr(err))
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
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'ctrl+right':
      self.config['tidx'] += 10
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'alt+right':
      self.config['tidx'] += 100
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'left':
      self.config['tidx'] -= 1
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'ctrl+left':
      self.config['tidx'] -= 10
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'alt+left':
      self.config['tidx'] -= 100
      Nt = self.data_set.shape[0]
      self.config['tidx'] = self.config['tidx']%Nt
      self.update()

    elif event.key == 'up':
      self.config['xidx'] += 1
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'ctrl+up':
      self.config['xidx'] += 10
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'alt+up':
      self.config['xidx'] += 100
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'down':
      self.config['xidx'] -= 1
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'ctrl+down':
      self.config['xidx'] -= 10
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 'alt+down':
      self.config['xidx'] -= 100
      Nx = self.data_set.shape[1]
      self.config['xidx'] = self.config['xidx']%Nx
      self.update()

    elif event.key == 's':
      if not os.path.exists('frames'):
        os.mkdir('frames')
        
      self.save_frames('frames')

    elif event.key == 'h':
      self.config['highlight'] = not self.config['highlight']
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

def interactive_strain_viewer(*args,**kwargs):
  ''' 
  wrapper to initiate and show an InteractiveStrainViewer
  '''
  iv = InteractiveStrainViewer(*args,**kwargs)
  iv.connect()
  plt.show()
  return

