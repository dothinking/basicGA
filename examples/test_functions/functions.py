# test functions for single-objective optimization cases
# 2-dimensional
# minimize

import numpy as np	

class FUN:
	def __init__(self):
		self.objective = None
		self.ranges = None
		self.solution = None

	@property
	def value(self):
		return self.objective(np.array(self.solution[0]))

	def plot(self, ax, comp=None, n=100):
		# Make data.
		X = np.linspace(self.ranges[0][0], self.ranges[0][1], n)
		Y = np.linspace(self.ranges[1][0], self.ranges[1][1], n)
		X, Y = np.meshgrid(X, Y)
		Z = self.objective(np.array([X,Y]))

		# Plot the surface.
		ax.contour(X, Y, Z)
		ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap='rainbow')
		ax.contour(X, Y, Z, offset=0, cmap='rainbow')

		# solution
		for x in self.solution:
			ax.scatter(x[0], x[1], self.value, c='r', marker='^')

		# comparison
		if comp and isinstance(comp, (tuple, list)) and len(comp)==3:
			ax.scatter(comp[0], comp[1], comp[2], c='b', marker='o')

class Ackley(FUN):
	def __init__(self):
		self.objective = lambda x: -20*np.exp(-0.2*(0.5*(x[0]**2+x[1]**2))**0.5)-np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))+np.exp(1)+20
		self.ranges = [(-5.0,5.0)] * 2
		self.solution = [(0.0,0.0)]

class Beale(FUN):
	def __init__(self):
		self.objective = lambda x: (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2
		self.ranges = [(-4.5,4.5)] * 2
		self.solution = [(3,0.5)]

class Booth(FUN):
	def __init__(self):
		self.objective = lambda x: (x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2
		self.ranges = [(-10,10)] * 2
		self.solution = [(1.0,3.0)]

class Bukin(FUN):
	def __init__(self):
		self.objective = lambda x: 100*np.abs(x[1]-x[0]**2/100)**0.5 + np.abs(x[0]+10)/100
		self.ranges = [(-15,-5.0), (-3,3)]
		self.solution = [(-10.0,1.0)]

class Easom(FUN):
	def __init__(self):
		self.objective = lambda x: -np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2)
		self.ranges = [(-100,100)] * 2
		self.solution = [(np.pi,np.pi)]

class Eggholder(FUN):
	def __init__(self):
		self.objective = lambda x: -(x[1]+47)*np.sin(np.abs(x[0]/2+x[1]+47)**0.5) - x[0]*np.sin(np.abs(x[0]-x[1]-47)**0.5)
		self.ranges = [(-512,512)] * 2
		self.solution = [(512,404.2319)]

class GoldsteinPrice(FUN):
	def __init__(self):
		self.objective = lambda x: (1+(1+x[0]+x[1])**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*\
			(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
		self.ranges = [(-2,2)] * 2
		self.solution = [(0.0,-1.0)]		

class Himmelblau(FUN):
	def __init__(self):
		self.objective = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
		self.ranges = [(-5,5)] * 2
		self.solution = [(3,2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]

class HolderTable(FUN):
	def __init__(self):
		self.objective = lambda x: -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-(x[0]**2+x[1]**2)**0.5/np.pi)))
		self.ranges = [(-10,10)] * 2
		self.solution = [(8.05502,9.66459),(-8.05502,9.66459),(8.05502,-9.66459),(-8.05502,-9.66459)]		

class Matyas(FUN):
	def __init__(self):
		self.objective = lambda x: 0.26*(x[0]**2+x[1]**2) - 0.48*x[0]*x[1]
		self.ranges = [(-10,10)] * 2
		self.solution = [(0.0,0.0)]

class McCormick(FUN):
	def __init__(self):
		self.objective = lambda x: np.sin(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1
		self.ranges = [(-1.5,4),(-3,4)]
		self.solution = [(-0.54719,-1.54719)]

class Rastrigin(FUN):
	def __init__(self):
		self.objective = lambda x: 20+x[0]**2+x[1]**2-10*np.cos(2.0*np.pi*x[0])-10*np.cos(2.0*np.pi*x[1])
		self.ranges = [(-5.12,5.12)] * 2
		self.solution = [(0.0,0.0)]		

class Rosenbrock(FUN):
	def __init__(self):
		self.objective = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
		self.ranges = [(-5.12,5.12)] * 2
		self.solution = [(1.0,1.0)]

class Schaffer(FUN):
	def __init__(self):
		self.objective = lambda x: 0.5 + (np.sin((x[0]**2+x[1]**2)**0.5)**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2
		self.ranges = [(-5,5)] * 2
		self.solution = [(0,0)]

class Schaffer_N4(FUN):
	def __init__(self):
		self.objective = lambda x: 0.5 + (np.cos(np.sin(np.abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2
		self.ranges = [(-25.0,25.0)] * 2
		self.solution = [(0,1.25313)]

class StyblinskiTang(FUN):
	def __init__(self):
		self.objective = lambda x: x[0]**4+x[1]**4-16*(x[0]**2+x[1]**2)+5*(x[0]+x[1])
		self.ranges = [(-5,5)] * 2
		self.solution = [(-2.903534,-2.903534)]



if __name__ == '__main__':

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	f = Bukin()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	f.plot(ax)
	plt.title(f.__class__.__name__)
	plt.show()
