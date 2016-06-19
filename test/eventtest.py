#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import rbf.halton
# my goal is the write a function that plots a row and column of a 
# matrix. I also want to be able to cycle through the rows and columns 
# with the arrow keys
''' 
fig,axs = plt.subplots(1,2)
M = np.random.random((10,10))
RIDX = [0] 
CIDX = [0]
axs[0].plot(M[RIDX[0],:])
axs[1].plot(M[:,CIDX[0]])

def onkey(event):
    print(vars(event))
    print(type(event.key))
    if event.key == 'right':
      CIDX[0] += 1
    elif event.key == 'left':
      CIDX[0] -= 1
    elif event.key == 'up':
      RIDX[0] += 1
    elif event.key == 'down':
      RIDX[0] -= 1

    print(CIDX)
    print(RIDX)
    axs[0].cla()
    axs[1].cla()
    axs[0].plot(M[RIDX[0],:])
    axs[1].plot(M[:,CIDX[0]])
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()
def _static_view(data,t,x,
                 tidx,xidx,
                 S,L,D,
                 highlight=True):
  ax1 = L.get_axes()
  ax1.set_title('station %s' % xidx)
  #ax2.scatter(x[:,0],x[:,1],c=data[tidx,:],s=20)
  #if highlight:
  #  ax2.plot(x[xidx,0],x[xidx,1],'ko',markersize=20)
  #return
  return 
'''

class InteractiveView:
  def __init__(self,data,t,x):
    fig1,ax1 = plt.subplots(3,1)
    fig2,ax2 = plt.subplots()
    self.tidx = 0
    self.xidx = 0
    self.highlight = True
    self.fig1 = fig1
    self.fig2 = fig2
    self.ax1 = ax1
    self.ax2 = ax2
    self.data = data
    self.t = t
    self.x = x

    self._init_draw()


  def connect(self):
    self.fig1.canvas.mpl_connect('key_press_event',self._onkey)
    self.fig2.canvas.mpl_connect('key_press_event',self._onkey)
    self.fig2.canvas.mpl_connect('pick_event',self._onpick)


  def _init_draw(self):
    self.ax1[0].set_title('station %s' % self.xidx)
    self.ax2.set_title('time %s' % self.tidx)

    self.S = self.ax2.quiver(self.x[:,0],self.x[:,1],
                             self.data[self.tidx,:,0],
                             self.data[self.tidx,:,1])

    self.L1, = self.ax1[0].plot(self.t,
                                self.data[:,self.xidx,0])

    self.L2, = self.ax1[1].plot(self.t,
                                self.data[:,self.xidx,1])

    self.L3, = self.ax1[2].plot(self.t,
                                self.data[:,self.xidx,2])

    self.D = self.ax2.plot(self.x[self.xidx,0],self.x[self.xidx,1],'ko',
                           markersize=10)[0]
    self.pickables = []
    for xi in self.x:
      self.pickables += self.ax2.plot(xi[0],xi[1],'.',picker=10,markersize=0)

    self.fig1.tight_layout()
    self.fig2.tight_layout()
    self.fig1.canvas.draw()
    self.fig2.canvas.draw()


  def _draw(self):
    self.tidx = self.tidx%self.data.shape[0]
    self.xidx = self.xidx%self.data.shape[1]

    self.ax1[0].set_title('station %s' % self.xidx)
    self.ax2.set_title('time %s' % self.tidx)
    self.S.set_UVC(self.data[self.tidx,:,0],
                   self.data[self.tidx,:,1])

    self.L1.set_data(self.t,
                     self.data[:,self.xidx,0])
    self.L2.set_data(self.t,
                     self.data[:,self.xidx,1])
    self.L3.set_data(self.t,
                     self.data[:,self.xidx,2])

    self.D.set_data(self.x[self.xidx,0],
                    self.x[self.xidx,1])
    if self.highlight:
      self.D.set_markersize(10)
    else:
      self.D.set_markersize(0)

    self.fig1.canvas.draw()
    self.fig2.canvas.draw()


  def _onpick(self,event):
    for i,v in enumerate(self.pickables):
      if event.artist == v:
        self.xidx = i
        break

    self._draw()    


  def _onkey(self,event):
    print(event.key)
    if event.key == 'right':
      self.tidx += 1
    if event.key == 'ctrl+right':
      self.tidx += 10

    elif event.key == 'left':
      self.tidx -= 1
    elif event.key == 'ctrl+left':
      self.tidx -= 10

    elif event.key == 'up':
      self.xidx += 1
    elif event.key == 'ctrl+up':
      self.xidx += 10

    elif event.key == 'down':
      self.xidx -= 1
    elif event.key == 'ctrl+down':
      self.xidx -= 10

    elif event.key == 'c':
      self.highlight = not self.highlight

    self._draw()    
    
Nt = 100
Nx = 1000
data = np.random.random((Nt,Nx,3))
t = np.linspace(0.0,1.0,Nt)
x = rbf.halton.halton(Nx,2)

a = InteractiveView(data,t,x)
a.connect()
plt.show()
quit()
#fig,axs = plt.subplots(1,2)
#_static_view(data,t,x,0,0,axs[0],axs[1])
#plt.show()
#quit()

class MatrixViewer:
  def __init__(self,M):
    self.ridx = 0
    self.cidx = 0
    fig,axs = plt.subplots(1,2)
    axs[0].plot(M[self.ridx,:])
    axs[1].plot(M[:,self.cidx])
    self.fig = fig
    self.axs = axs
    self.M = M

  def connect(self):
    self.fig.canvas.mpl_connect('key_press_event',self.onkey)
    self.fig.canvas.mpl_connect('pick_event',self.onpick)
 
  def draw(self):
    self.axs[0].cla()
    self.axs[1].cla()
    self.axs[0].plot(self.M[self.ridx,:],picker=10)
    self.axs[1].plot(self.M[:,self.cidx],picker=10)
    self.fig.canvas.draw()
    
  def onpick(self,event):
    print('foo!!!')

  def onkey(self,event):
    if event.key == 'right':
      self.cidx += 1
    elif event.key == 'left':
      self.cidx -= 1
    elif event.key == 'up':
      self.ridx += 1
    elif event.key == 'down':
      self.ridx -= 1

    self.draw()    
    
        
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

M = np.random.random((20,10))
A = MatrixViewer(M)
A.connect()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#a#x.set_title('click to build line segments')
#line, = ax.plot([0], [0])  # empty line
#linebuilder = LineBuilder(line)

plt.show()
