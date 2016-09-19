''' 
This module provides functions for plotting strain
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from matplotlib.collections import PatchCollection
from pygeons.view import disable_default_key_bindings
from pygeons.view import without_interactivity
from pygeons._input import restricted_input
from math import sin,cos,sqrt


def strain_glyph(x,y,grad,sigma=None)
                 scale=1.0,
                 extension_color='b',
                 compression_color='r',
                 alpha=0.2,
                 k=100):
  ''' 
  Returns the artists making up a two-dimensional strain glyph which 
  indicates the magnitude of normal strain in any direction. The 
  normal strain pointing in the direction of n is defined as
  
    n.dot(E).dot(n)
  
  where n is a unit vector and E is the infinitesimal strain tensor 
  with components
  
    E = [[e_xx,e_xy],[e_xy,eyy]].
   
  If the normal strain is positive then it indicates extension in that 
  direction, otherwise it indicates compression. The strain glyph 
  consists of each point 
  
    n*(n.dot(E).dot(n)) 
  
  for n pointing in directions ranging from 0 to 2*pi.  The points are 
  connected making up a petal shaped glyph, where each petal has a 
  color indicating extensional or compressional normal strain.
  
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

    k : int, optional
      number of vertices used in the glyph. Higher numbers result in 
      higher resolution

  Returns
  -------
    out : tuple
      artists making up the strain glyph

  '''
  # convert deformation gradient to strain 
  exx = grad[0],
  eyy = grad[3]
  exy = 0.5*(grad[1] + grad[2])
  sxx = sigma[0]
  syy = sigma[3]
  sxy = sqrt(0.25*sigma[1]**2 + 0.25*sigma[2]**2)

  mean_ext = []
  mean_cmp = []
  ub_ext = []
  ub_cmp = []
  lb_ext = []
  lb_cmp = []

  theta = np.linspace(0.0,2*np.pi,k)
  for t in theta:
    # normal vector rotated about the origin by t
    n = [cos(t),sin(t)]
    # maps strain to the normal strain component 
    norm = (exx*n[0]**2 +
            eyy*n[1]**2 +
            exy*2*n[0]*n[1])*scale
    # uncertainty in the normal strain component
    norm_sigma = sqrt(sxx**2*n[0]**4 +
                      syy**2*n[1]**4 +
                      sxy**2*(2*n[0]*n[1])**2)*scale

    if norm >= 0.0:
      mean_ext += [[n[0]*norm,
                    n[1]*norm]]
    else:
      mean_cmp += [[n[0]*norm,
                    n[1]*norm]]

    if (norm+norm_sigma) >= 0.0:
      ub_ext += [[n[0]*(norm + norm_sigma),
                  n[1]*(norm + norm_sigma)]]
    else:
      ub_cmp += [[n[0]*(norm + norm_sigma),
                  n[1]*(norm + norm_sigma)]]

    if (norm-norm_sigma) >= 0.0:
      lb_ext += [[n[0]*(norm - norm_sigma),
                  n[1]*(norm - norm_sigma)]]
    else:
      lb_cmp += [[n[0]*(norm - norm_sigma),
                  n[1]*(norm - norm_sigma)]]
      

  mean_ext = np.array(mean_ext).reshape(-1,2)
  mean_cmp = np.array(mean_cmp).reshape(-1,2)
  ub_ext = np.array(ub_ext).reshape(-1,2)
  lb_ext = np.array(lb_ext).reshape(-1,2)
  ub_cmp = np.array(ub_cmp).reshape(-1,2)
  lb_cmp = np.array(lb_cmp).reshape(-1,2)

  out = ()
  # draw the extensional component of the uncertainty 
  ext_poly_x = x + np.concatenate((ub_ext[:,0],lb_ext[::-1,0]))
  ext_poly_y = y + np.concatenate((ub_ext[:,1],lb_ext[::-1,1]))
  ext_poly_coo = np.array([ext_poly_x,ext_poly_y]).T
  if ext_poly_coo.shape[0] != 0:
    out += Polygon(ext_poly_coo,facecolor=extension_color,
                   edgecolor='none',alpha=alpha),

  # draw the compressional component of the uncertainty 
  cmp_poly_x = x + np.concatenate((ub_cmp[:,0],lb_cmp[::-1,0]))
  cmp_poly_y = y + np.concatenate((ub_cmp[:,1],lb_cmp[::-1,1]))
  cmp_poly_coo = np.array([cmp_poly_x,cmp_poly_y]).T
  if cmp_poly_coo.shape[0] != 0:
    out += Polygon(cmp_poly_coo,facecolor=compression_color,
                   edgecolor='none',alpha=alpha),

  # draw the solid line indicating the mean strain
  out += Line2D(x + mean_ext[:,0],y + mean_ext[:,1],color=extension_color),
  out += Line2D(x + mean_cmp[:,0],y + mean_cmp[:,1],color=compression_color),
  return out


def _principle_strain_component(x,y,eigval,eigvec,scale,**kwargs):
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


def principle_strain_glyph(x,y,dudx,dudy,dvdx,dvdy,scale=1.0,**kwargs):
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
  artists += _principle_component(x,y,eigvals[0],eigvecs[:,0],scale,**kwargs)
  artists += _principle_component(x,y,eigvals[1],eigvecs[:,1],scale,**kwargs)
  return artists


class Strain:
  def __init__(self,x,y,
               ux,uy,vx,vy,
               sux=None,suy=None,
               svx=None,svy=None,
               **kwargs):
    N = len(x)
    if sux is None: sux = np.zeros(N)
    if suy is None: suy = np.zeros(N)
    if svx is None: svx = np.zeros(N)
    if svy is None: svy = np.zeros(N)
      
    artists = ()
    for args in zip(x,y,
                    ux,uy,vx,vy,
                    sux,suy,svx,svy): 
      artists += strain_glyph(*args,**kwargs)

    self.x = x
    self.y = y
    self.kwargs = kwargs
    self.artists = artists

  def set_gradient(self,
                   ux,uy,vx,vy,
                   sux=None,suy=None,
                   svx=None,svy=None):
    N = len(self.x)
    if sux is None: sux = np.zeros(N)
    if suy is None: suy = np.zeros(N)
    if svx is None: svx = np.zeros(N)
    if svy is None: svy = np.zeros(N)

    artists = ()
    for args in zip(self.x,self.y,
                    ux,uy,vx,vy,
                    sux,suy,svx,svy): 
      artists += strain_glyph(*args,**self.kwargs)
    
    self.artists = artists
    
  def get_artists(self):
    return self.artists
    
  def remove(self):
    for a in self.artists: a.remove()  
      

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
        
      ux,uy,vx,vy : (Nt,Nx) array
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
                    self.sigma_set[self.config['tidx'],:,0],
                    self.sigma_set[self.config['tidx'],:,1],
                    self.sigma_set[self.config['tidx'],:,2],
                    self.sigma_set[self.config['tidx'],:,3],
                    scale=self.config['scale'])

    for a in self.strain.get_artists(): self.ax.add_artist(a)
    self.ax.autoscale_view()

  def _update_strain(self):
    self.strain.remove()
    self.strain.set_gradient(
      self.data_set[self.config['tidx'],:,0],
      self.data_set[self.config['tidx'],:,1],
      self.data_set[self.config['tidx'],:,2],
      self.data_set[self.config['tidx'],:,3],
      self.sigma_set[self.config['tidx'],:,0],
      self.sigma_set[self.config['tidx'],:,1],
      self.sigma_set[self.config['tidx'],:,2],
      self.sigma_set[self.config['tidx'],:,3])

    for a in self.strain.get_artists(): self.ax.add_artist(a)

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

