''' 
This module provides functions for plotting strain
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.text import Text
from pygeons.view import disable_default_key_bindings
from pygeons.view import without_interactivity
from pygeons.view import one_sigfig
from pygeons._input import restricted_input
from scipy.spatial import cKDTree

def strain_glyph(x,y,grad,sigma=None,
                 scale=1.0,
                 extension_color='b',
                 compression_color='r',
                 alpha=0.2,
                 vertices=500):
  ''' 
  Returns the artists making up a two-dimensional strain glyph which 
  indicates the magnitude of normal strain in any direction. The 
  normal strain pointing in the direction of *n* is defined as
  
    n.dot(eps).dot(n)
  
  where *n* is a unit vector and *eps* is the infinitesimal strain 
  tensor with components
  
    eps = [[          du/dx,  (dv/dx+du/dy)/2],
           [(dv/dx+du/dy)/2,            dv/dy]].
   
  If the normal strain is positive then it indicates extension in the 
  *n* direction, otherwise it indicates compression. The strain glyph 
  consists of each point
  
    pos + n*(n.dot(E).dot(n)) 
  
  for *n* pointing in directions ranging from 0 to 2*pi, where *pos* 
  is the vector [*x*,*y*]. The points are connected making up a petal 
  shaped glyph, where each petal has a color indicating extensional or 
  compressional normal strain.
  
  If uncertainties are provided then the 68% confidence interval is 
  also shown in the glyph. Regions of the confidence interval which 
  are extensional and compressional are colored accordingly.

  Parameters
  ----------
    x,y : float
      center of the glyph
      
    grad : (4,) array
      list of deformation gradient components. The order is du/dx, 
      du/dy, dv/dx, dv/dy

    sigma : (4,) array
      corresponding uncertainties in the deformation gradient 
      components
      
    extension_color : optional
      string or tuple indicating the color used for extensional normal 
      strain

    compresion_color : optional
      string or tuple indicating the color used for compresional 
      normal strain

    alpha : float, optional
      transparency of the confidence interval

    vertices : int, optional
      number of vertices used in the glyph. Higher numbers result in 
      higher resolution

  Returns
  -------
    out : tuple
      artists making up the strain glyph

  '''
  if sigma is None: sigma = np.zeros(4)
  
  # convert deformation gradient to strain 
  exx,eyy,exy =  grad[0],  grad[3],                      0.5*(grad[1] + grad[2])
  sxx,syy,sxy = sigma[0], sigma[3], np.sqrt(0.25*sigma[1]**2 + 0.25*sigma[2]**2)

  theta = np.linspace(0.0,2*np.pi,vertices)
  # (k,2) array of normal vectors pointing in the direction of theta
  n = np.array([np.cos(theta),np.sin(theta)]).T
  # (k,) array of normal strains 
  norm = (exx*n[:,0]**2 +
          eyy*n[:,1]**2 +
          exy*2*n[:,0]*n[:,1])*scale
  # (k,) array of normal strain uncertainties
  norm_sigma = np.sqrt(sxx**2*n[:,0]**4 +
                       syy**2*n[:,1]**4 +
                       sxy**2*(2*n[:,0]*n[:,1])**2)*scale
  # (k,2) array of vectors in the theta direction with length equal to 
  # the normal strain
  mean = n*norm[:,None]
  mean_ext = mean[norm>=0.0]
  mean_cmp = mean[norm<0.0]

  # upper and lower bound on the vector length
  ub = n*(norm[:,None] + norm_sigma[:,None])  
  ub_ext = ub[(norm+norm_sigma)>=0.0]
  ub_cmp = ub[(norm+norm_sigma)<0.0]

  lb = n*(norm[:,None] - norm_sigma[:,None])  
  lb_ext = lb[(norm-norm_sigma)>=0.0]
  lb_cmp = lb[(norm-norm_sigma)<0.0]
      
  out = []
  # draw the extensional component of the uncertainty 
  ext_poly_x = x + np.concatenate((ub_ext[:,0],lb_ext[::-1,0]))
  ext_poly_y = y + np.concatenate((ub_ext[:,1],lb_ext[::-1,1]))
  ext_poly_coo = np.array([ext_poly_x,ext_poly_y]).T
  if ext_poly_coo.shape[0] != 0:
    out += [Polygon(ext_poly_coo,facecolor=extension_color,
                    edgecolor='none',alpha=alpha)]

  # draw the compressional component of the uncertainty 
  cmp_poly_x = x + np.concatenate((ub_cmp[:,0],lb_cmp[::-1,0]))
  cmp_poly_y = y + np.concatenate((ub_cmp[:,1],lb_cmp[::-1,1]))
  cmp_poly_coo = np.array([cmp_poly_x,cmp_poly_y]).T
  if cmp_poly_coo.shape[0] != 0:
    out += [Polygon(cmp_poly_coo,facecolor=compression_color,
                    edgecolor='none',alpha=alpha)]

  # draw the solid line indicating the mean strain
  out += [Line2D(x + mean_ext[:,0],y + mean_ext[:,1],color=extension_color)]
  out += [Line2D(x + mean_cmp[:,0],y + mean_cmp[:,1],color=compression_color)]
  return out


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
               ux,uy,vx,vy,
               sux=None,suy=None,
               svx=None,svy=None,
               scale=None,
               units=None,
               time_labels=None,
               fontsize=10,
               ax=None,
               title=None,
               ylim=None,
               xlim=None,
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
        
      ux,uy,vx,vy : (Nt,Nx) array
        Deformation gradient components

      sux,suy,svx,svy : (Nt,Nx) array
        Uncertainty of the gradient components
        
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
    
    if sux is None: sux = np.zeros((Nt,Nx))
    if suy is None: suy = np.zeros((Nt,Nx))
    if svx is None: svx = np.zeros((Nt,Nx))
    if svy is None: svy = np.zeros((Nt,Nx))

    tpl = (ux[:,:,None],uy[:,:,None],vx[:,:,None],vy[:,:,None])
    self.data_set = np.concatenate(tpl,axis=2)

    tpl = (sux[:,:,None],suy[:,:,None],svx[:,:,None],svy[:,:,None])
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
      mag = 5*np.nanmean(np.abs([ux,uy,vx,vy]))
      mag = max(mag,1e-10)
      # find the average distance between points
      T = cKDTree(self.x) 
      if Nx <= 1:
        dist = 1.0
      else:
        dist = np.mean(T.query(self.x,2)[0][:,1])
    
      scale = dist/mag
      
    if key_magnitude is None:  
      mag = 3*np.nanmean(np.abs([ux,uy,vx,vy]))
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

  def _remove_artists(self):
    while len(self.artists) > 0:
      self.artists.pop().remove()

  def _draw_key(self):
    mag = self.config['key_magnitude']
    units = self.config['units']
    grad = [mag,0.0,0.0,-mag]
    sigma = [0.1*mag,0.0,0.0,-0.1*mag]

    key_pos_axes = self.config['key_position']
    key_pos_display = self.ax.transAxes.transform(key_pos_axes)
    key_pos_data = self.ax.transData.inverted().transform(key_pos_display)
    posx,posy = key_pos_data
    self.artists += strain_glyph(posx,posy,grad,sigma,
                                 scale=self.config['scale'],
                                 extension_color=self.config['extension_color'],
                                 compression_color=self.config['compression_color'],
                                 alpha=self.config['alpha'],
                                 vertices=self.config['vertices'])
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
                          color=self.config['compression_color'])]
  
  def _draw_strain(self): 
    grad = self.data_set[self.config['tidx'],:,:]
    sigma = self.sigma_set[self.config['tidx'],:,:]
    for args in zip(self.x[:,0],self.x[:,1],grad,sigma):
      self.artists += strain_glyph(*args,
                                   scale=self.config['scale'],
                                   extension_color=self.config['extension_color'],
                                   compression_color=self.config['compression_color'],
                                   alpha=self.config['alpha'],
                                   vertices=self.config['vertices'])
       
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

