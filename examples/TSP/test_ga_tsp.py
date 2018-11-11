import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

# built-in modules from GA Package
from GA.GAPopulation.SequenceIndividual import UniqueLoopIndividual
from GA.GAPopulation.Population import Population
from GA.GAOperators.Selection import RouletteWheelSelection, LinearRankingSelection
from GA.GAOperators.Crossover import UniqueSeqCrossover
from GA.GAOperators.Mutation import UniqueSeqMutation
from GA.GAProcess import GA

from TspCities import TSPCities


class TSPPopulation(Population):
	'''user defined Poopulation for TSP'''
	def __init__(self, cities, individual, size=50):
		super().__init__(individual, size)
		self.cities = cities

	def initialize(self):
		'''initialization for next generation'''
		super().initialize()

		for I in self.individuals:
			if np.random.rand()<0.5:
				I.solution = self._nearest_neighbor_path()

	def _nearest_neighbor_path(self):
		D = self.cities.distances.copy()
		D[D==0] = np.inf		

		solution = np.zeros_like(self.individual.solution)
		solution[0] = np.random.randint(self.individual.dimension)
		D[solution[0],:] = np.inf

		for i in np.arange(1, self.individual.dimension):
			_id = D[:, solution[i-1]].argmin()			
			solution[i] = _id
			D[_id,:] = np.inf
		return solution

	def _two_opt(self, solution, fun_evaluation):
	    count = 0		
	    while count < 50:
	    	pos = np.random.choice(self.individual.dimension, 2)
	    	start, end = pos.min(), pos.max()

	    	new_solution = solution.copy()
	    	new_solution[start:end+1] = solution[start:end+1][::-1] # reverse genes at specified positions

	    	if fun_evaluation(new_solution)>=fun_evaluation(solution):
	    		count += 1
	    	else:
	    		count = 0
	    		solution = new_solution
	    return solution

	def evaluate(self, fun_evaluation, fun_fitness):
		'''
		calculate objectibe value and fitness for each individual.
		fun_evaluation	: objective function
		fun_fitness  	: population fitness based on evaluation
		'''

		# get the value directly if it has been calculated before
		for I in self.individuals:
			I.solution = self._two_opt(I.solution, fun_evaluation)

		evaluation = np.array([fun_evaluation(I.solution) for I in self.individuals])	

		# calculate fitness
		fitness = fun_fitness(evaluation)
		fitness = fitness/fitness.sum() # normalize

		# set attributes for each individual
		for I, e, f in zip(self.individuals, evaluation, fitness):
			I.evaluation = e
			I.fitness = f

		return fitness, evaluation


	def best0(self, fun_evaluation, fun_fitness):
		'''
		get best individual according to evaluation value,
		then optimize it by 2-opt method
		'''
		the_best = super().best(fun_evaluation, fun_fitness)
		the_best.solution = self._two_opt(the_best.solution.copy(), fun_evaluation)
		the_best.evaluation = fun_evaluation(the_best.solution)
		fitness = fun_fitness(the_best.evaluation)
		the_best.fun_fitness = fitness/fitness.sum() # normalize
		return the_best

def test(cities, gen):

	# GA process
	I = UniqueLoopIndividual(cities.dimension)
	P = TSPPopulation(cities, I, 50)
	L = LinearRankingSelection()
	R = RouletteWheelSelection()
	C = UniqueSeqCrossover([0.75, 0.95])
	M = UniqueSeqMutation(0.2)
	g = GA(P, R, C, M)

	# solve
	res = g.run(cities.distance, gen)

	return res


if __name__ == '__main__':

	cities = TSPCities('dataset/eil51.tsp', 'dataset/eil51.opt.tour')

	# build-in GA process
	s1 = time.time()
	res = test(cities, 500)

	# output
	s2 = time.time()
	print('Global solution      : {0}'.format(cities.min_distance))
	print('TSP based GA solution: {0} in {1} seconds'.format(res.evaluation, s2-s1))

	# plot
	cities.plot_cities(plt)
	cities.plot_path(plt, cities.solution)
	cities.plot_path(plt, res.solution)
	plt.show()