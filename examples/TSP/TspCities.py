import numpy as np

class TSPCities:
	"""TSP data set"""
	def __init__(self, filename, sol_filename=None):
		arr = np.loadtxt(filename)
		self.cities = arr[:,1:]
		self.dimension = arr.shape[0]
		self.solution = np.loadtxt(sol_filename).astype(np.int32)-1 if sol_filename else None
		self.distances = self.init_distances()

	def init_distances(self):
		X = self.cities[:,0].reshape((1,self.dimension))
		Y = self.cities[:,1].reshape((1,self.dimension))
		d = ((X-X.T)**2 + (Y-Y.T)**2)**0.5
		pos = np.arange(self.dimension)
		d[pos,pos] = np.inf
		return d

	def distance(self, tour):
		temp_tour = np.empty_like(tour)
		temp_tour[0:-1] = tour[1:]
		temp_tour[-1] = tour[0]
		res = self.distances[tour, temp_tour]
		return res.sum()


	@property
	def min_distance(self):
		return self.distance(self.solution) if not self.solution is None else None

	def plot_cities(self, plt):
		for i,city in enumerate(self.cities):
			plt.scatter(city[0], city[1], c='b', marker='o')
			plt.annotate(str(i), xy=(city[0], city[1]))

	def plot_path(self, plt, tour):
		if tour is None: return
		if not self.solution is None and not any(tour-self.solution):
			ls = '--'
		else:
			ls = '-'
		data = self.cities[tour,:]
		plt.plot(data[:,0], data[:,1], linestyle=ls)




if __name__ == '__main__':

	import matplotlib.pyplot as plt

	T = TSPCities('dataset/a280.tsp', 'dataset/a280.opt.tour')
	print(T.min_distance)

	T.plot_cities(plt)
	T.plot_path(plt, T.solution)
	plt.show()
