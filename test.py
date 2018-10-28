# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator, FormatStrFormatter

import numpy as np

from GA import GA

import time
# ------------------
# testing functions
# ------------------
# [-2,1]
f1 = lambda x: x[0]*np.sin(10*np.pi*x[0])+2

# sol: x=[0,0], min=0
f2 = lambda x : 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))

# schaffer
# sol: x=[0,0], min=0
schaffer = lambda x: 0.5 + (np.sin((x[0]**2+x[1]**2)**0.5)**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# schaffer-N4
# sol: x=[0,1.25313], min=0.292579
schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(np.abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# shubert
# sol: x=[], min=
def shubert(x):
	a = [i*np.cos((i+1)*x[0]+i) for i in range(1,6)]	 
	b = [i*np.cos((i+1)*x[1]+i) for i in range(1,6)]
	time.sleep(0.00005)
	return sum(a) * sum(b)


# Rosenbrock: -5.12<=x<=5.12
# sol: x=[1,1], min=0
rosenbrock = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2

# ------------------
# plots
# ------------------
def surface_plot(f, lbound, ubound, ax, n=200):
	# Make data.
	X = np.linspace(lbound[0], ubound[0], n)
	Y = np.linspace(lbound[1], ubound[1], n)
	X, Y = np.meshgrid(X, Y)
	Z = f([X,Y])

	# Plot the surface.
	# res = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.contourf(X, Y, Z, cmap=cm.PuBu_r)

# ------------------
# test GA
# ------------------
def test(obj, sol):

	# theoretical res
	print('\nGlobal solution input: {0}'.format(sol))
	print('Global solution output: {0}\n'.format(obj(sol)))

	# GA res
	kw = {
		'size'	: 100,
		'max_generation': 50,
		# 'fitness': lambda x: np.exp(-x),
		'selection_mode': 'elite',
		# 'selection_elite': False,
		'crossover_rate': [0.5, 0.9],
		# 'crossover_rate': 0.8,
		'crossover_alpha': 0.5,
		'mutation_rate'	: 0.08
	}
	lb = [-10,-10]
	ub = [10, 10]
	g = GA(obj, 2, lb, ub, **kw)
	I = g.solve()

	print('\nGA solution input: {0}'.format(I.chrom))
	print('GA solution output: {0}\n'.format(I.evaluation))

	# plots
	ax = plt.subplot(111)
	surface_plot(obj, lb, ub, ax)
	ax.scatter(I.chrom[0], I.chrom[1], c='r', marker='o')
	ax.scatter(sol[0], sol[1], c='b', marker='^')

	plt.show()


if __name__ == '__main__':	

	# test(f1, [0,0])
	# test(f2, [0,0])
	# test(schaffer, [0,0])
	# test(schaffer_n4, [0,1.25313])
	# test(shubert, [0,0])
	test(rosenbrock, [1,1])