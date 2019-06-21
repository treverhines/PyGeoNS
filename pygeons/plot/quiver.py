import numpy as np

from matplotlib.quiver import Quiver
from matplotlib.collections import EllipseCollection
from matplotlib.text import Text
from scipy.spatial import cKDTree
import warnings


def _estimate_scale(x, y, u, v):
  pos = np.array([x, y]).T
  # return a scale of 1 if there is only one datum
  if pos.shape[0] == 0:
    return 1.0

  T = cKDTree(pos)
  average_dist = np.mean(T.query(pos, 2)[0][:, 1])
  average_length = np.mean(np.sqrt(u**2 + v**2))
  return average_length/average_dist
                   

class QuiverWithUncertainty:
  def __init__(self, ax, x, y, u, v, 
               su=None, sv=None, scale=None, 
               include_key=True,
               key_pos=(0.1, 0.1), key_mag=1.0, key_label='1 unit',
               ellipse_kwargs=None, quiver_kwargs=None, text_kwargs=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if su is None:
      su = np.zeros_like(u)
    else:
      su = np.asarray(su, dtype=float)
                
    if sv is None:
      sv = np.zeros_like(v)
    else:
      sv = np.asarray(sv, dtype=float)

    if quiver_kwargs is None:
        quiver_kwargs = {}

    if 'scale_units' in quiver_kwargs:
      raise ValueError('"scale_units" is fixed at "xy" for `Quiver`')         

    if 'angles' in quiver_kwargs:
      raise ValueError('"angles" is fixed at "xy" for `Quiver`')         

    if 'pivot' in quiver_kwargs:
      raise ValueError('"pivot" is fixed at "tail" for `Quiver`')         
    
    if ellipse_kwargs is None:
        ellipse_kwargs = {'edgecolors':'k',
                          'facecolors':'none',
                          'linewidths':1.0}

    if 'offsets' in ellipse_kwargs:
      raise ValueError('"offsets" cannot be set for `EllipseCollection`')

    if 'units' in ellipse_kwargs:
      raise ValueError('"units" is fixed at "xy" for `EllipseCollection`')
    
    if scale is None:
      scale = _estimate_scale(x, y, u, v)

    if text_kwargs is None:
      text_kwargs = {}      

    self.ax = ax
    self.x = x
    self.y = y
    self.u = u
    self.v = v
    self.su = su
    self.sv = sv
    self.scale = scale
    self.quiver_kwargs = quiver_kwargs
    self.ellipse_kwargs = ellipse_kwargs
    self.text_kwargs = text_kwargs
    self.include_key = include_key
    self.key_pos = key_pos
    self.key_mag = key_mag
    self.key_label = key_label
    self._init_quiver()
    self._init_quiver_key()
    self._init_ellipses()

  def _init_ellipses(self):
    centers_x = self.x + self.u/self.scale
    centers_y = self.y + self.v/self.scale
    centers = np.array([centers_x, centers_y]).T
    width = 2.0*self.su/self.scale
    height = 2.0*self.sv/self.scale
    # do not draw ellipses which are too small relative to the vector length
    too_small = 0.001
    length = np.sqrt((self.u/self.scale)**2 + (self.v/self.scale)**2)
    with warnings.catch_warnings():
      # do not print out zero division warning
      warnings.simplefilter("ignore")
      is_not_too_small = ((np.nan_to_num(width/length) > too_small) |
                          (np.nan_to_num(height/length) > too_small))

    width = width[is_not_too_small]
    height = height[is_not_too_small]
    centers = centers[is_not_too_small]
    angle = np.zeros_like(width)
    
    # dont add ellipses if there are no ellipses to add
    if any(is_not_too_small):
      self.ellipses = EllipseCollection(width, height, angle,
                                        units='xy',
                                        offsets=centers,
                                        transOffset=self.ax.transData,
                                        **self.ellipse_kwargs)
      self.ax.add_artist(self.ellipses)
    else:
      self.ellipses = None

  def _update_ellipses(self):
    if self.ellipses is not None:
      self.ellipses.remove()
    
    self._init_ellipses()
  
  def _init_quiver(self):
    self.quiver = Quiver(self.ax, self.x, self.y, self.u, self.v,
                         scale=self.scale,
                         scale_units='xy',
                         angles='xy',
                         pivot='tail',
                         **self.quiver_kwargs)
    self.ax.add_artist(self.quiver)

  def _init_quiver_key(self):
    if not self.include_key:
      self.text = None
      self.quiver_key = None
      
    else:    
      self.quiver_key = Quiver(self.ax, 
                               [self.key_pos[0]], [self.key_pos[1]],
                               [self.key_mag], [0.0],
                               scale=self.scale,
                               scale_units='xy',
                               angles='xy',
                               pivot='tail',
                               transform=self.ax.transAxes,
                               **self.quiver_kwargs)
      self.text = Text(self.key_pos[0], self.key_pos[1] + 0.03, self.key_label,
                       transform=self.ax.transAxes, **self.text_kwargs)
      self.ax.add_artist(self.quiver_key)
      self.ax.add_artist(self.text)
        
  def _update_quiver(self):
    self.quiver.set_UVC(self.u, self.v)

  def set_data(self, u, v, su=None, sv=None):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if su is None:
      su = np.zeros_like(u)
    else:
      su = np.asarray(su, dtype=float)

    if sv is None:
      sv = np.zeros_like(v)
    else:
      sv = np.asarray(sv, dtype=float)

    self.u = u
    self.v = v
    self.su = su
    self.sv = sv
    self._update_quiver()
    self._update_ellipses()

  def remove(self):
    self.quiver.remove()
    if self.text is not None:
      self.text.remove()

    if self.quiver_key is not None:  
      self.quiver_key.remove()
      
    if self.ellipses is not None:
      self.ellipses.remove()
