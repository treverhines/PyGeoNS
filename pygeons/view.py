#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import rbf.halton
import matplotlib
from rbf.interpolant import RBFInterpolant
from matplotlib.cm import ScalarMappable
import pygeons.quiver
import rbf.basis
import modest

# change behavior of mpl.quiver. this is necessary for error 
# ellipses but may lead to insidious bugs... 
matplotlib.quiver.Quiver = pygeons.quiver.Quiver

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


@modest.funtime
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
                     order=1,basis=rbf.basis.phs1)
  uitp = I(pnts_itp)
  uitp = uitp.reshape((x.shape[0],y.shape[0]))                   
  return uitp
  

class InteractiveView:
  def __init__(self,data_sets,t,x,
               sigma_sets=None,
               cmap=None,
               quiver_key_label=None,
               quiver_key_length=1.0,
               quiver_scale=10.0,
               quiver_key_pos=None,
               station_names=None,
               data_set_names=None,
               vmin=None,
               vmax=None,
               ylim=None,
               xlim=None,
               time_series_axs=None,
               time_series_title=None,
               map_ax=None,
               map_title=None,
               fontsize=10,
               ylabel='displacement [m]',
               xlabel='time [years]',
               clabel='vertical displacement [m]'):
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
        
      quiver_key_label : str

      quiver_key_length : float

      quiver_scale : float

      quiver_key_pos : (2,) array
        
      station_names : (Nx,) str array
      
      data_set_names : (Ns,) str array
      
      vmin : float
      
      vmax : float
      
      time_series_axs : (3,) list of Axis instances
        list of three axes where time series will be plotted

      map_ax : Axis instance
        axis where map view will be plotted
      
      ylabel : str
        time series y label
      
      xlabel : str
        time series x label
        
      clabel : str
        color bar label  
      
    '''
    if time_series_axs is None:
      fig1,ax1 = plt.subplots(3,1,sharex=True)
      self.fig1 = fig1
      self.ax1 = ax1
    else:
      self.fig1 = time_series_axs[0].get_figure()
      self.ax1 = time_series_axs
      
    if map_ax is None:
      fig2,ax2 = plt.subplots()
      self.fig2 = fig2
      self.ax2 = ax2
    else:
      self.fig2 = map_ax.get_figure()  
      self.ax2 = map_ax
      
    self.highlight = True
    self.tidx,self.xidx = 0,0
    if sigma_sets is None:
      sigma_sets = [np.ones(d.shape) for d in data_sets]

    self.data_sets = []
    self.sigma_sets = []
    for d,s in zip(data_sets,sigma_sets):      
      dout,sout = _make_masked_array(d,s)
      self.data_sets += [dout]
      self.sigma_sets += [sout]

    self.t = t
    self.x = x
    self.cmap = cmap
    self.vmin = vmin
    self.vmax = vmax
    self.ylim = ylim
    self.xlim = xlim
    self.quiver_scale = quiver_scale
    self.xlabel = xlabel # xlabel for time series plot
    self.ylabel = ylabel # ylabel for time series plots
    self.clabel = clabel
    self.color_cycle = ['k',(0.0,0.7,0.0),'r','g','c','m','y']
    if station_names is None:
      station_names = np.arange(len(self.x)).astype(str)
    if data_set_names is None:
      data_set_names = np.arange(len(self.data_sets)).astype(str)

    self.station_names = list(station_names)
    self.data_set_names = list(data_set_names)
    
    if quiver_key_pos is None:
      quiver_key_pos = (0.2,0.1)

    if quiver_key_label is None:   
      quiver_key_label = str(quiver_key_length) + ' [m]'

    self.quiver_key_pos = quiver_key_pos
    self.quiver_key_label = quiver_key_label
    self.quiver_key_length = quiver_key_length
    self.time_series_title = time_series_title
    self.map_title = map_title
    self.fontsize = fontsize

    self._init_draw()

  def connect(self):
    self.fig1.canvas.mpl_connect('key_press_event',self._onkey)
    self.fig2.canvas.mpl_connect('key_press_event',self._onkey)
    self.fig2.canvas.mpl_connect('pick_event',self._onpick)


  def _init_draw(self):
    ''' 
      creates the following artists
        D : marker for current station
        P : station pickers
        L1-L3 : list of time series for each data_set
        F1-F3 : list of fill between series for each data_set
        Q : list of quiver instances for each data_set
        K : quiver key for first element in Q
        I : interpolated image of z component for first data_set
        S : scatter plot where color is z component for second data_set

    '''
    self.ax1[2].set_xlabel(self.xlabel)
    self.ax1[0].set_ylabel(self.ylabel)
    self.ax1[1].set_ylabel(self.ylabel)
    self.ax1[2].set_ylabel(self.ylabel)
    # dont convert to exponential form
    self.ax1[0].get_xaxis().get_major_formatter().set_useOffset(False)
    self.ax1[1].get_xaxis().get_major_formatter().set_useOffset(False)
    self.ax1[2].get_xaxis().get_major_formatter().set_useOffset(False)

    if self.time_series_title is None:
      self.ax1[0].set_title('station %s' % self.station_names[self.xidx],fontsize=self.fontsize)
    else:
      self.ax1[0].set_title(self.time_series_title,fontsize=self.fontsize)

    if self.map_title is None:
      self.ax2.set_title('time %g' % self.t[self.tidx],fontsize=self.fontsize)
    else:
      self.ax2.set_title(self.map_title,fontsize=self.fontsize)
      
    self.ax2.set_aspect('equal')
    self.ax2.title.set_fontsize(self.fontsize)

    self.ax1[0].xaxis.label.set_fontsize(self.fontsize)
    self.ax1[1].xaxis.label.set_fontsize(self.fontsize)
    self.ax1[2].xaxis.label.set_fontsize(self.fontsize)
    self.ax1[0].yaxis.label.set_fontsize(self.fontsize)
    self.ax1[1].yaxis.label.set_fontsize(self.fontsize)
    self.ax1[2].yaxis.label.set_fontsize(self.fontsize)
    
    self.ax1[0].title.set_fontsize(self.fontsize)
    self.ax1[0].tick_params(labelsize=self.fontsize)
    self.ax1[1].tick_params(labelsize=self.fontsize)
    self.ax1[2].tick_params(labelsize=self.fontsize)
    self.ax2.tick_params(labelsize=self.fontsize)

    # highlighted point
    self.D = self.ax2.plot(self.x[self.xidx,0],
                           self.x[self.xidx,1],'ko',
                           markersize=20*self.highlight)[0]
    # pickable artists
    self.P = []
    for xi in self.x:
      self.P += self.ax2.plot(xi[0],xi[1],'.',
                              picker=10,
                              markersize=0)

    # if ylim and xlim were not user specified then fix them now
    if self.xlim is None:
      self.xlim = self.ax2.get_xlim()

    if self.ylim is None:  
      self.ylim = self.ax2.get_ylim()
    
    self.Q = []
    self.L1,self.L2,self.L3 = [],[],[]
    self.F1,self.F2,self.F3 = [],[],[]
    for si in range(len(self.data_sets)):
      self.Q += [self.ax2.quiver(self.x[:,0],self.x[:,1],
                        self.data_sets[si][self.tidx,:,0],
                        self.data_sets[si][self.tidx,:,1],
                        scale=self.quiver_scale,  
                        width=0.004,
                        sigma=(self.sigma_sets[si][self.tidx,:,0],
                               self.sigma_sets[si][self.tidx,:,1],
                               0.0*self.sigma_sets[si][self.tidx,:,0]),
                        color=self.color_cycle[si],
                        ellipse_edgecolors=self.color_cycle[si],
                        zorder=2+si)]

      # time series instances
      self.L1 += self.ax1[0].plot(self.t,
                                  self.data_sets[si][:,self.xidx,0],
                                  color=self.color_cycle[si],
                                  label=self.data_set_names[si])
      self.F1 += [self.ax1[0].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,0] -
                                  self.sigma_sets[si][:,self.xidx,0],
                                  self.data_sets[si][:,self.xidx,0] +
                                  self.sigma_sets[si][:,self.xidx,0],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      self.L2 += self.ax1[1].plot(self.t,
                                  self.data_sets[si][:,self.xidx,1],
                                  color=self.color_cycle[si],
                                  label=self.data_set_names[si])
      self.F2 += [self.ax1[1].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,1] -
                                  self.sigma_sets[si][:,self.xidx,1],
                                  self.data_sets[si][:,self.xidx,1] +
                                  self.sigma_sets[si][:,self.xidx,1],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      self.L3 += self.ax1[2].plot(self.t,
                                  self.data_sets[si][:,self.xidx,2],
                                  color=self.color_cycle[si],
                                  label=self.data_set_names[si])
      self.F3 += [self.ax1[2].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,2] -
                                  self.sigma_sets[si][:,self.xidx,2],
                                  self.data_sets[si][:,self.xidx,2] +
                                  self.sigma_sets[si][:,self.xidx,2],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      # quiver key
      if si == 0:
        self.K = self.ax2.quiverkey(self.Q[si],
                                    self.quiver_key_pos[0],
                                    self.quiver_key_pos[1],
                                    self.quiver_key_length,
                                    self.quiver_key_label,zorder=2,
                                    labelsep=0.05,
                                    fontproperties={'size':self.fontsize})

      if si == 0:
        # interpolate z value for first data set
        self.x_itp = [np.linspace(self.xlim[0],self.xlim[1],100),
                      np.linspace(self.ylim[0],self.ylim[1],100)]
        data_itp = _grid_interp_data(self.data_sets[si][self.tidx,:,2],
                                     self.x,self.x_itp[0],self.x_itp[1])
        
        if self.vmin is None:
          # self.vmin and self.vmax are the user specified color 
          # bounds. if they are None then the color bounds will be 
          # updated each time the artists are redrawn
          vmin = data_itp.min()
        else:  
          vmin = self.vmin

        if self.vmax is None:
          vmax = data_itp.max()
        else:
          vmax = self.vmax
          
        self.I = self.ax2.imshow(data_itp,extent=(self.xlim+self.ylim),
                                 interpolation='bicubic',
                                 origin='lower',
                                 vmin=vmin,vmax=vmax,
                                 cmap=self.cmap,zorder=0)
        self.I.set_clim((vmin,vmax))

        self.cbar = self.fig2.colorbar(self.I)  
        self.cbar.set_clim((vmin,vmax))
        self.cbar.set_label(self.clabel,fontsize=self.fontsize)
        self.cbar.ax.tick_params(labelsize=self.fontsize)
        self.cbar.solids.set_rasterized(True)

      if si == 1:  
        sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cmap)
        # use scatter points to show z for second data set 
        colors = sm.to_rgba(self.data_sets[si][self.tidx,:,2])
        self.S = self.ax2.scatter(self.x[:,0],self.x[:,1],
                                  c=colors,
                                  s=50,zorder=1,
                                  edgecolor=self.color_cycle[si])
      
    self.ax2.set_ylim(self.ylim)
    self.ax2.set_xlim(self.xlim)
    self.ax1[0].legend(frameon=False,fontsize=self.fontsize)
    self.ax1[0].set_autoscale_on(True) 
    self.ax1[1].set_autoscale_on(True) 
    self.ax1[2].set_autoscale_on(True) 
    self.ax1[0].relim()
    self.ax1[1].relim()
    self.ax1[2].relim()
    self.ax1[0].autoscale_view()
    self.ax1[1].autoscale_view()
    self.ax1[2].autoscale_view()
    self.fig1.tight_layout()
    #self.fig2.tight_layout()
    self.fig1.canvas.draw()
    self.fig2.canvas.draw()


  def _draw(self):
    self.tidx = self.tidx%self.data_sets[0].shape[0]
    self.xidx = self.xidx%self.data_sets[0].shape[1]

    if self.time_series_title is None:
      self.ax1[0].set_title('station %s' % self.station_names[self.xidx],fontsize=self.fontsize)
    else:
      self.ax1[0].set_title(self.time_series_title,fontsize=self.fontsize)
      
    if self.map_title is None:
      self.ax2.set_title('time %g' % self.t[self.tidx],fontsize=self.fontsize)
    else:
      self.ax2.set_title(self.map_title,fontsize=self.fontsize)
      
    self.ax2.set_aspect('equal')

    self.D.set_data(self.x[self.xidx,0],
                    self.x[self.xidx,1])
    self.D.set_markersize(20*self.highlight)

    # unfortunately there is no option to update the fill between instances
    # and so they must be deleted and redrawn
    [self.ax1[0].collections.remove(f) for f in self.F1]
    [self.ax1[1].collections.remove(f) for f in self.F2]
    [self.ax1[2].collections.remove(f) for f in self.F3]
    self.F1,self.F2,self.F3 = [],[],[]
      
    for si in range(len(self.data_sets)):
      self.Q[si].set_UVC(self.data_sets[si][self.tidx,:,0],
                         self.data_sets[si][self.tidx,:,1],
                         sigma=(self.sigma_sets[si][self.tidx,:,0],
                                self.sigma_sets[si][self.tidx,:,1],
                                0.0*self.sigma_sets[si][self.tidx,:,0]))

      self.L1[si].set_data(self.t,
                           self.data_sets[si][:,self.xidx,0])
      # relabel in case the data_set order has switched
      self.L1[si].set_label(self.data_set_names[si])                     
      
      self.F1 += [self.ax1[0].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,0] -
                                  self.sigma_sets[si][:,self.xidx,0],
                                  self.data_sets[si][:,self.xidx,0] +
                                  self.sigma_sets[si][:,self.xidx,0],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      self.L2[si].set_data(self.t,
                           self.data_sets[si][:,self.xidx,1])
      self.L2[si].set_label(self.data_set_names[si])                     

      self.F2 += [self.ax1[1].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,1] -
                                  self.sigma_sets[si][:,self.xidx,1],
                                  self.data_sets[si][:,self.xidx,1] +
                                  self.sigma_sets[si][:,self.xidx,1],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      self.L3[si].set_data(self.t,
                           self.data_sets[si][:,self.xidx,2])
      self.L3[si].set_label(self.data_set_names[si])                     

      self.F3 += [self.ax1[2].fill_between(self.t,
                                  self.data_sets[si][:,self.xidx,2] -
                                  self.sigma_sets[si][:,self.xidx,2],
                                  self.data_sets[si][:,self.xidx,2] +
                                  self.sigma_sets[si][:,self.xidx,2],
                                  edgecolor='none',
                                  color=self.color_cycle[si],alpha=0.5)]

      if si == 0:
        data_itp = _grid_interp_data(self.data_sets[si][self.tidx,:,2],
                                     self.x,
                                     self.x_itp[0],self.x_itp[1])
        self.I.set_data(data_itp)

        if self.vmin is None:
          # self.vmin and self.vmax are the user specified color 
          # bounds. if they are None then the color bounds will be 
          # updated each time the artists are redrawn
          vmin = data_itp.min()
        else:  
          vmin = self.vmin

        if self.vmax is None:
          vmax = data_itp.max()
        else:
          vmax = self.vmax

        self.I.set_clim((vmin,vmax))
        self.cbar.set_clim((vmin,vmax))
        self.cbar.solids.set_rasterized(True)

      if si == 1:
        # set new colors for scatter plot. use the same colormap as 
        # the one used for data set 1
        sm = ScalarMappable(norm=self.cbar.norm,cmap=self.cmap)
        colors = sm.to_rgba(self.data_sets[si][self.tidx,:,2])
        self.S.set_facecolors(colors)
        
    # fix the axes for ax2 and make sure the axes for ax1 are automatic
    self.ax2.set_ylim(self.ylim)
    self.ax2.set_xlim(self.xlim)
    self.ax1[0].legend(frameon=False,fontsize=self.fontsize)
    self.ax1[0].set_autoscale_on(True) 
    self.ax1[1].set_autoscale_on(True) 
    self.ax1[2].set_autoscale_on(True) 
    self.ax1[0].relim()
    self.ax1[1].relim()
    self.ax1[2].relim()
    self.ax1[0].autoscale_view()
    self.ax1[1].autoscale_view()
    self.ax1[2].autoscale_view()

    self.fig1.canvas.draw()
    self.fig2.canvas.draw()


  def _onpick(self,event):
    for i,v in enumerate(self.P):
      if event.artist == v:
        self.xidx = i
        break

    self._draw()    


  def _onkey(self,event):
    if event.key == 'right':
      self.tidx += 1

    elif event.key == 'ctrl+right':
      self.tidx += 10

    elif event.key == 'alt+right':
      self.tidx += 100

    elif event.key == 'left':
      self.tidx -= 1

    elif event.key == 'ctrl+left':
      self.tidx -= 10

    elif event.key == 'alt+left':
      self.tidx -= 100

    elif event.key == 'up':
      self.xidx += 1

    elif event.key == 'ctrl+up':
      self.xidx += 10

    elif event.key == 'alt+up':
      self.xidx += 100

    elif event.key == 'down':
      self.xidx -= 1

    elif event.key == 'ctrl+down':
      self.xidx -= 10

    elif event.key == 'alt+down':
      self.xidx -= 100

    elif event.key == 'c':
      self.highlight = not self.highlight

    elif event.key == 'r':
      # roll order of data arrays 
      self.data_sets = _roll(self.data_sets)
      self.data_set_names = _roll(self.data_set_names)
      self.sigma_sets = _roll(self.sigma_sets)

    else:
      # do nothing
      return

    self._draw()    
    
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
    raise ValueError('specified values of u, v, z, su, sv, or sz must have the same length')
      
  data_sets = []
  sigma_sets = []
  for i in range(Ns):
    data_sets += [np.concatenate((u[i][:,:,None],v[i][:,:,None],z[i][:,:,None]),axis=2)]
    sigma_sets += [np.concatenate((su[i][:,:,None],sv[i][:,:,None],sz[i][:,:,None]),axis=2)]
  
  iv = InteractiveView(data_sets,t,x,sigma_sets=sigma_sets,**kwargs)
  iv.connect()
  plt.show()
    

if __name__ == '__main__':
  Nt = 10
  Nx = 50
  t = 2010 + np.linspace(0.0,1.0,Nt)
  x = rbf.halton.halton(Nx,2)
  data = (np.cos(2*np.pi*t[:,None,None]) *
          np.sin(2*np.pi*x[:,0])[None,:,None] *
          np.cos(2*np.pi*x[:,1])[None,:,None])
  data = data.repeat(3,axis=2)
  data[:,:,[0,1]] = 0.0

  network_viewer(t,x,z=[data[:,:,2],data[:,:,0]],vmin=-1,vmax=1)
  quit()