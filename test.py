# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator, FormatStrFormatter

import math
import numpy as np

from GA import GA


# ------------------
# testing functions
# ------------------
# [-2,1]
f1 = lambda x: x[0]*math.sin(10*np.pi*x[0])+2

# sol: x=[0,0], min=0
f2 = lambda x : 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*np.pi*x[0])+math.cos(2*np.pi*x[1]))

# schaffer
# sol: x=[0,0], min=0
schaffer = lambda x: 0.5 + (math.sin((x[0]**2+x[1]**2)**0.5)**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# schaffer-N4
# sol: x=[0,1.25313], min=0.292579
schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(np.abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# shubert
# sol: x=[], min=
def shubert(x):
	a = [i*math.cos((i+1)*x[0]+i) for i in range(1,6)]	 
	b = [i*math.cos((i+1)*x[1]+i) for i in range(1,6)]
	return sum(a) * sum(b)


# Rosenbrock: -5.12<=x<=5.12
# sol: x=[1,1], min=0
rosenbrock = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2

# ------------------
# plots
# ------------------
def surface_plot(f, lbound, ubound, n=200):
	fig, ax = plt.subplots()

	# Make data.
	X = np.linspace(lbound[0], ubound[0], n)
	Y = np.linspace(lbound[1], ubound[1], n)
	X, Y = np.meshgrid(X, Y)
	Z = f([X,Y])

	# Plot the surface.
	# res = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	res = ax.contourf(X, Y, Z, cmap=cm.PuBu_r)

	# Add a color bar which maps values to colors.
	fig.colorbar(res, shrink=0.5, aspect=10)

	return ax


# ------------------
# test GA
# ------------------
def test(obj, sol):
	# GA res
	kw = {
		'lbound': [-10, -10],
		'ubound': [10, 10],
		'size'	: 195,
		'max_generation': 30,
		'fitness': lambda x: np.exp(-x),
		'selection_mode': 'elite',
		'crossover_rate': 0.9,		
		'crossover_alpha': 0.01,
		'mutation_rate'	: 0.1,
	}
	g = GA(obj, 2, **kw)
	P = g.solve()

	# theoretical res
	res = P.best[1]
	res0 = obj(sol)
	print('\nGlobal solution input: {0}'.format(sol))
	print('Global solution output: {0}\n'.format(res0))

	# print('Relative error: {0} %'.format((res-res0)/res0*100))

	# plots
	ax = surface_plot(schaffer_n4, kw['lbound'], kw['ubound'])
	x = [I.chrom[0] for I in P.individuals]
	y = [I.chrom[1] for I in P.individuals]
	ax.scatter(x,y,c='r',marker='o')
	plt.show()


if __name__ == '__main__':	

	# test(f2, [0,0])
	# test(schaffer, [0,0])
	# test(schaffer_n4, [0,1.25313])
	# test(schaffer, [0,0])
	test(rosenbrock, [1,1])