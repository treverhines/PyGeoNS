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

Controls
--------
    Enter : edit the configurable parameters through the command line.
        Variables can be defined using any functions in the numpy, 
        matplotlib, or base python namespace

    Left : move back 1 time step (Ctrl-Left and Alt-Left move back 10 
        and 100 respectively)

    Right : move forward 1 time step (Ctrl-Right and Alt-Right move 
        forward 10 and 100 respectively)

    R : redraw figures

Notes
-----
    Exit PISV by closing the figures
  
    Key bindings only work when the active window is one of the PISV 
    figures   

---------------------------------------------------------------------
  '''
  def __init__(self,t,x,
               exx,eyy,exy,
               sxx=None,syy=None,sxy=None,
               scale=None,
               units=None,
               time_labels=None,
               fontsize=10,
               ax=None,
               title=None,
               ylim=None,
               xlim=None,
               contraction_color='r',
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
         
      ax : Axis instance
        Axis where map view will be plotted

      title : str
        Replaces the default title for the map view plot

      ylim : (2,) array
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
    if ax is None:
      # gives a white background 
      fig,ax = plt.subplots(num='Map View',facecolor='white')
      self.fig = fig
      self.ax = ax
    else:
      self.fig = ax.get_figure()
      self.ax = ax

    if time_labels is None:
      time_labels = np.array(self.t).astype(str)

    self.time_labels = list(time_labels)

    # estimate a good scale for the strain glyphs
    if scale is None:
      # Get an idea of what the typical strain magnitudes are
      mag = 5*np.nanmean(np.abs([exx,eyy,exy]))
      mag = max(mag,1e-10)
      # find the average distance between points
      T = cKDTree(self.x) 
      if Nx <= 1:
        dist = 1.0
      else:
        dist = np.mean(T.query(self.x,2)[0][:,1])
    
      scale = dist/mag
      
    if key_magnitude is None:  
      mag = 3*np.nanmean(np.abs([exx,eyy,exy]))
      mag = max(mag,1e-10)
      key_magnitude = one_sigfig(mag)
      
    # this dictionary contains plot configuration parameters which may 
    # be interactively changed
    self.config = {}
    self.config['tidx'] = 0
    self.config['scale'] = scale
    self.config['title'] = title
    self.config['xlim'] = xlim
    self.config['ylim'] = ylim
    self.config['fontsize'] = fontsize
    self.config['units'] = units
    self.config['contraction_color'] = contraction_color
    self.config['extension_color'] = extension_color
    self.config['alpha'] = alpha
    self.config['vertices'] = vertices
    self.config['key_magnitude'] = key_magnitude
    self.config['key_position'] = key_position
    self._init()
    disable_default_key_bindings()
    print(self.__doc__)

  def connect(self):
    self.fig.canvas.mpl_connect('key_press_event',self.on_key_press)

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

  def _init_ax(self):
    # call after _init_scatter
    self.ax.set_aspect('equal')
    self.ax.tick_params(labelsize=self.config['fontsize'])
    if self.config['title'] is None:
      time_label = self.time_labels[self.config['tidx']]
      self.ax.set_title('time : %s' % time_label,
                        fontsize=self.config['fontsize'])
    else:
      self.ax.set_title(self.config['title'],
                        fontsize=self.config['fontsize'])

    # do not dynamically update the axis limits
    if self.config['xlim'] is None:
      self.config['xlim'] = self.ax.get_xlim()

    if self.config['ylim'] is None:
      self.config['ylim'] = self.ax.get_ylim()

    self.ax.set_xlim(self.config['xlim'])
    self.ax.set_ylim(self.config['ylim'])

  def _update_ax(self):
    if self.config['title'] is None:
      time_label = self.time_labels[self.config['tidx']]
      self.ax.set_title('time : %s' % time_label,
                        fontsize=self.config['fontsize'])
    else:
      self.ax.set_title(self.config['title'],
                        fontsize=self.config['fontsize'])

  def _remove_artists(self):
    while len(self.artists) > 0:
      self.artists.pop().remove()

  def _draw_key(self):
    mag = self.config['key_magnitude']
    units = self.config['units']
    strain = [mag,-mag,0.0]
    sigma = [0.25*mag,0.25*mag,0.25*mag]

    key_pos_axes = self.config['key_position']
    key_pos_display = self.ax.transAxes.transform(key_pos_axes)
    key_pos_data = self.ax.transData.inverted().transform(key_pos_display)
    posx,posy = key_pos_data
    self.artists += strain_glyph(key_pos_data,strain,sigma=sigma,
                                 scale=self.config['scale'],
                                 ext_color=self.config['extension_color'],
                                 cnt_color=self.config['contraction_color'],
                                 alpha=self.config['alpha'],
                                 vert=self.config['vertices'])
    if units is None:
      text_str = '%s' % mag
    else:
      text_str = '%s %s' % (mag,units)
      
    textx = posx + 1.1*mag*self.config['scale']
    texty = posy
    self.artists += [Text(textx,texty,text_str,
                          fontsize=10,
                          color=self.config['extension_color'])]
    textx = posx
    texty = posy + 1.1*mag*self.config['scale']
    self.artists += [Text(textx,texty,'-' + text_str,
                          fontsize=10,
                          color=self.config['contraction_color'])]
  
  def _draw_strain(self): 
    strain = self.data_set[self.config['tidx'],:,:]
    sigma = self.sigma_set[self.config['tidx'],:,:]
    for args in zip(self.x,strain,sigma):
      self.artists += strain_glyph(*args,
                                   scale=self.config['scale'],
                                   ext_color=self.config['extension_color'],
                                   cnt_color=self.config['contraction_color'],
                                   alpha=self.config['alpha'],
                                   vert=self.config['vertices'])
       
    for a in self.artists: self.ax.add_artist(a)

  def _init(self):
    self.artists = []
    self._init_ax()
    self._draw_key()
    self._draw_strain()
    self.fig.canvas.draw()
    self.ax.autoscale_view()

  def update(self):
    self._remove_artists()
    self._update_ax()
    self._draw_key()
    self._draw_strain()
    self.fig.canvas.draw()

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

    elif event.key == 's':
      if not os.path.exists('frames'):
        os.mkdir('frames')
        
      self.save_frames('frames')

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

