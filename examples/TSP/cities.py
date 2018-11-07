import numpy as np

class TSP:
	"""TSP data set"""
	def __init__(self, filename, sol_filename=None):
		arr = np.loadtxt(filename)
		self.cities = arr[:,1:]
		self.dimension = arr.shape[0]
		self.solution = np.loadtxt(sol_filename).astype(np.int32)-1 if sol_filename else None

	def distances(self, tour):
		temp_tour = np.empty_like(tour)
		temp_tour[0:-1] = tour[1:]
		temp_tour[-1] = tour[0]
		d = (self.cities[tour,:]-self.cities[temp_tour,:])**2
		res = np.sqrt(d.sum(axis=1))
		return res.sum()

	@property
	def min_distance(self):
		return self.distances(self.solution) if not self.solution is None else None

	def plot_cities(self, ax):
		for city in self.cities:
			ax.scatter(city[0], city[1], c='b', marker='o')

	def plot_path(self, ax, tour):
		if not tour is None:
			data = self.cities[tour,:]
			ax.plot(data[:,0], data[:,1])




if __name__ == '__main__':

	import matplotlib.pyplot as plt
	

	T = TSP('dataset/eil51.tsp', 'dataset/eil51.opt.tour')
	tour = np.random.choice(T.dimension, T.dimension, replace=False)

	print(T.distances(tour))
	print(T.min_distance)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	T.plot_cities(ax)
	T.plot_path(ax, T.solution)
	T.plot_path(ax, tour)
	plt.show()




