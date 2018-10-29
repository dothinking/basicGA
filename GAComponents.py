#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np

class Individual:
	'''individual of population'''
	def __init__(self, ranges):
		'''
		ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
		validation of ranges is skipped...
		'''		
		self.ranges = np.array(ranges)
		self.dimension = self.ranges.shape[0]

		# initialize solution within [lb, ub]
		seeds = np.random.random(self.dimension)
		lb = self.ranges[:, 0]
		ub = self.ranges[:, 1]
		self._solution = lb + (ub-lb)*seeds

		# evaluation and fitness
		self.evaluation = None
		self.fitness = None

	@property
	def solution(self):
		return self._solution

	@solution.setter
	def solution(self, solution):
		assert self.dimension == solution.shape[0]
		assert (solution>=self.ranges[:, 0]).all() and (solution<=self.ranges[:, 1]).all()
		self._solution = solution
	

class Population:
	'''collection of individuals'''
	def __init__(self, individual, size=50):
		'''
		individual   : individual template
		size         : count of individuals		
		'''
		self.individual = individual
		self.size = size
		self.individuals = None

	def initialize(self):
		'''initialization for next generation'''
		IndvClass = self.individual.__class__
		self.individuals = np.array([IndvClass(self.individual.ranges) for i in range(self.size)], dtype=IndvClass)

	def best(self, fun_evaluation, fun_fitness):
		'''get best individual according to evaluation value '''
		# evaluate first and collect evaluations
		_, evaluation = self.fitness(fun_evaluation, fun_fitness)
		# get the minimum position
		pos = np.argmin(evaluation)
		return self.individuals[pos]

	def fitness(self, fun_evaluation, fun_fitness, fun_adaptive=None):
		'''
		calculate objectibe value and fitness for each individual.
		fun_evaluation	: objective function
		fun_fitness  	: population fitness based on evaluation
		fun_adaptive	: the difference of fitness decrease with the increase of generation,
							which could not show the competition of domanit individual,
							so a adaptive function is used to enlarge the deviation
		'''

		# get the value directly if it has been calculated before
		evaluation = np.array([fun_evaluation(I.solution) if I.evaluation is None else I.evaluation for I in self.individuals])

		# calculate fitness
		fitness = fun_fitness(evaluation)
		fitness = fitness/np.sum(fitness) # normalize

		# a adaptive function is used to enlarge the deviation
		if fun_adaptive:
			fitness = fun_adaptive(fitness)
			fitness = fitness/np.sum(fitness) # normalize
		
		# set attributes for each individual	
		for I, e, f in zip(self.individuals, evaluation, fitness):
			I.evaluation = e
			I.fitness = f

		return fitness, evaluation
		

if __name__ == '__main__':

	ranges = [(-10,10)] * 3
	obj = lambda x: x[0]+x[1]**2+x[2]**3

	I = Individual(ranges)
	P = Population(I, 10)
	P.initialize()

	print(P.best(obj).solution)