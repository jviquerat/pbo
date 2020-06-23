from pylab import *
from matplotlib import ticker

def f(x,y):
    return x**2 + y**2
    #return (1.0-x)**2 + 100.0*(y-x**2)**2

xmin =-2.0
xmax = 2.0
ymin =-2.0
ymax = 2.0

n    = 600
lvls = [0.5, 1.0, 2.0, 3.0, 4.0]
x    = np.linspace(xmin,xmax,n)
y    = np.linspace(ymin,ymax,n)
X,Y  = np.meshgrid(x,y)

axes([0.0, 0.0, 1.0, 1.0])
contourf(X, Y, f(X,Y), lvls, alpha=0.9,
         locator=ticker.LogLocator(), cmap='RdBu',
         vmin=0.5, vmax=4.0)
C = contour(X, Y, f(X,Y), lvls, colors='black')
#clabel(C, inline=1, fontsize=8)
plt.gca().set_aspect("equal")
xticks([]), yticks([])
savefig('plot.png',dpi=200)
