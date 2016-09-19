''' 
This module provides functions for plotting strain
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.collections import PatchCollection
from pygeons.view import disable_default_key_bindings
from pygeons.view import without_interactivity
from pygeons._input import restricted_input


def _component(x,y,eigval,eigvec,scale,**kwargs):
  ''' 
  returns two arrow patches corresponding to a principle strain 
  component
  '''
  if eigval == 0.0:
    return ()
  elif eigval < 0.0:
    compressional = True
  else:
    compressional = False

  u = eigval*eigvec[0]*scale
  v = eigval*eigvec[1]*scale
  length = np.sqrt(u**2 + v**2)

  # set arrow parameters
  head_length = 0.25*length
  head_width = 0.5*head_length
  width = 0.03*length
  overhang = 0.05
  if compressional:
    arrow1 = FancyArrow(x-u,y-v,u,v,
                        length_includes_head=True,
                        width=width,
                        head_width=head_width,
                        head_length=head_length,
                        overhang=overhang,
                        **kwargs)
    arrow2 = FancyArrow(x+u,y+v,-u,-v,
                        length_includes_head=True,
                        width=width,
                        head_width=head_width,
                        head_length=head_length,
                        overhang=overhang,
                        **kwargs)
  else:
    arrow1 = FancyArrow(x,y,u,v,
                        length_includes_head=True,
                        width=width,
                        head_width=head_width,
                        head_length=head_length,
                        overhang=overhang,
                        **kwargs)
    arrow2 = FancyArrow(x,y,-u,-v,
                        length_includes_head=True,
                        width=width,
                        head_width=head_width,
                        head_length=head_length,
                        overhang=overhang,
                        **kwargs)

  return arrow1,arrow2


def get_principle_strain_artists(x,y,dudx,dudy,dvdx,dvdy,scale=1.0,**kwargs):
  ''' 
  returns the arrows patches corresponding to each principle strain 
  component
  '''
  strain = np.array([[dudx,0.5*(dudy + dvdx)],
                     [0.5*(dudy + dvdx),dvdy]])
  if np.any(np.isnan(strain)):
    return ()

  eigvals,eigvecs = np.linalg.eig(strain)
  artists = ()
  artists += _component(x,y,eigvals[0],eigvecs[:,0],scale,**kwargs)
  artists += _component(x,y,eigvals[1],eigvecs[:,1],scale,**kwargs)
  return artists


class Strain(PatchCollection):
  def __init__(self,x,y,dudx,dudy,dvdx,dvdy,scale=1.0,**kwargs):
    patches = ()
    for args in zip(x,y,dudx,dudy,dvdx,dvdy): 
      patches += get_principle_strain_artists(*args,scale=scale)

    PatchCollection.__init__(self,patches,**kwargs)
    self.x = x
    self.y = y
    self.scale=scale

  def set_gradient(self,dudx,dudy,dvdx,dvdy):
    patches = ()
    for args in zip(self.x,self.y,dudx,dudy,dvdx,dvdy): 
      patches += get_principle_strain_artists(*args,scale=self.scale)
    
    self.set_paths(patches)


class InteractiveStrainViewer:
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
               dudx,dudy,
               dvdx,dvdy,
               scale=1.0,
               time_labels=None,
               fontsize=10,
               ax=None,
               title=None,
               ylim=None,
               xlim=None):
    ''' 
    interactively views strain which is time and space dependent
    
    Parameters
    ----------
      t : (Nt,) array
        Observation times

      x : (Nx,2) array
        Observation positions
        
      dudx,dudy,dvdx,dvdy : (Nx,Nt) array
        Gradient of the vector field
        
      scale : float
        Increases the size of the strain markers
         
      ax : Axis instance
        axis where map view will be plotted

      title : str
        replaces the default title for the map view plot

      ylim : (2,) array
        ylim for the map view plot
      
      xlim : (2,) array
        xlim for the map view plot
      
      fontsize : float
        controls all fontsizes
        
    '''
    # time and space arrays
    self.t = np.asarray(t)
    self.x = np.asarray(x)

    tpl = (dudx[:,:,None],dudy[:,:,None],dvdx[:,:,None],dvdy[:,:,None])
    self.data_set = np.concatenate(tpl,axis=2)

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

    # this dictionary contains plot configuration parameters which may 
    # be interactively changed
    self.config = {}
    self.config['tidx'] = 0
    self.config['scale'] = scale
    self.config['title'] = title
    self.config['xlim'] = xlim
    self.config['ylim'] = ylim
    self.config['fontsize'] = fontsize
    self._init()
    disable_default_key_bindings()
    print(self.__doc__)

  def connect(self):
    self.fig.canvas.mpl_connect('key_press_event',self.on_key_press)

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

  def _init_strain(self):
    self.strain = Strain(self.x[:,0],self.x[:,1],
                    self.data_set[self.config['tidx'],:,0],
                    self.data_set[self.config['tidx'],:,1],
                    self.data_set[self.config['tidx'],:,2],
                    self.data_set[self.config['tidx'],:,3],
                    scale=self.config['scale'],
                    facecolor='k',
                    edgecolor='k',
                    zorder=2)

    self.ax.add_collection(self.strain,autolim=True)
    self.ax.autoscale_view()

  def _update_strain(self):
    self.strain.set_gradient(
      self.data_set[self.config['tidx'],:,0],
      self.data_set[self.config['tidx'],:,1],
      self.data_set[self.config['tidx'],:,2],
      self.data_set[self.config['tidx'],:,3])

  def _init(self):
    self._init_ax()
    self._init_strain()
    self.fig.canvas.draw()

  def update(self):
    self._update_ax()
    self._update_strain()
    self.fig.canvas.draw()

  def hard_update(self):
    # clears all axes and redraws everything
    self.strain.remove()
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

