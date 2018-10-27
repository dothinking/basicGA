#----------------------------------------------------------
# Population Object for GA
#----------------------------------------------------------
import numpy as np
import copy
from GAIndividual import Individual

class Population():
	'''collection of individuals'''
	def __init__(self, fun_evaluation, fun_fitness, size=50):
		'''
		fun_evaluation: objective function with x, e.g. lambda x: x[0]**2+x[1]
		fun_fitness   : definition to evaluate the objective values
		size          : count of individuals
		'''
		self._fun_evaluation = fun_evaluation
		self._fun_fitness = fun_fitness
		self._size = size
		self._individuals = None

	def initialize(self, dim, lbound, ubound):
		'''initialization for next generation'''
		self._evaluation = None
		self._individuals = np.array([Individual(dim, lbound, ubound) for i in range(self._size)])
		for individual in self._individuals:
			individual.initialize()	

	@property
	def individuals(self):
		return self._individuals

	@property
	def best(self):
		'''get best individual according to evaluation value '''
		# evaluate first and collect evaluations
		self._evaluate()
		evaluation = []
		for I in self._individuals:
			evaluation.append(I.evaluation)

		# get the minimum position
		pos = np.argmin(np.array(evaluation))
		return self._individuals[pos]

	def _evaluate(self):
		# calculate values according to the objective function
		evaluation = np.array([self._fun_evaluation(I.chrom) for I in self._individuals])

		# calculate fitness according to fitness function
		# sum(abs(V))-Vi by default
		fitness = self._fun_fitness(evaluation)		
		fitness = fitness/np.sum(fitness) # normalize
		
		# set attributes for each individual	
		for I,e,f in zip(self._individuals, evaluation, fitness):
			I.evaluation = e
			I.fitness = f

		return fitness

	def _select_by_roulette(self, fitness):
		'''
		select individuals by Roulette:
		individuals selected with a probability of its fitness
		'''		
		return np.random.choice(self._individuals, self._size, p=fitness)		

	def _select_by_elite(self, fitness):
		'''the top individuals will always be selected
		'''
		pos = np.argsort(fitness)
		num = max(int(self._size/3.0), 10) # first 1/3 individuals
		elite_pos = pos[self._size-num:] # asc order
		return np.random.choice(self._individuals[elite_pos], self._size)

	@staticmethod
	def _cross_individuals(individual_a, individual_b, alpha):
		'''
		generate two child individuals based on parent individuals:
		new values are calculated at random positions
		alpha: linear ratio to cross two genes, exchange two genes if alpha is 0.0
		'''
		# random positions to be crossed
		pos = np.random.rand(individual_a.dimension) <= 0.5

		# cross value
		new_value_a = individual_a.chrom*pos*alpha + individual_b.chrom*pos*(1-alpha)
		new_value_b = individual_a.chrom*pos*(1-alpha) + individual_b.chrom*pos*alpha

		# return new individuals
		new_individual_a = Individual(individual_a.dimension, *individual_a.bound)
		new_individual_b = Individual(individual_b.dimension, *individual_b.bound)

		new_individual_a.chrom = new_value_a
		new_individual_b.chrom = new_value_b

		return new_individual_a, new_individual_b

	def select(self, method='roulette'):
		'''select individuals'''

		# evaluate each individual first
		fitness = self._evaluate()

		# selection mode
		methods = {
			'roulette': self._select_by_roulette,
			'elite'  : self._select_by_elite
		}
		selected_individuals = methods.get(method, self._select_by_roulette)(fitness)

		# pay attention to deep copy these objects
		self._individuals = np.array([copy.deepcopy(I) for I in selected_individuals])

	def crossover(self, rate, alpha):
		'''crossover operation:
		rate: propability of crossover. adaptive rate for each individual if rate=-1
		alpha: factor for crossing two chroms
		'''
		new_individuals = []		
		random_population = np.random.permutation(self._individuals) # random order
		num = int(self._size/2.0)+1
		for individual_a, individual_b in zip(self._individuals[0:num+1], random_population[0:num+1]):
			if np.random.rand() <= rate:
				child_individuals = self._cross_individuals(individual_a, individual_b, alpha)
				new_individuals.extend(child_individuals)
			else:
				new_individuals.append(individual_a)
				new_individuals.append(individual_b)

		self._individuals = np.array(new_individuals[0:self._size+1])

	def mutate(self, num, rate, alpha):
		'''
		mutation operation:
		num: count of mutating gene
		rate: propability of mutation. adaptive if rate=-1
		alpha: mutating magnitude
		'''
		for individual in self._individuals:
			if np.random.rand() <= rate:
				pos = np.random.choice(individual.dimension, num, replace=False)
				individual.mutate(pos, alpha)


if __name__ == '__main__':

	# create Population
	fun_obj     = lambda x: x[0]**2-1
	fun_fitness = lambda x: np.exp(-x)
	# fun_fitness = lambda x: np.sum(np.abs(x))-x

	P = Population(fun_obj, fun_fitness, 10)

	# initialize individuals
	dimension = 6
	lbound = np.array([-10]*dimension)
	ubound = np.array([10]*dimension)
	P.initialize(dimension, lbound, ubound)

	print('Before Operation:')
	for I in P.individuals:
		print(I.chrom.tolist())

	# operation
	P.select()
	P.crossover(0.7, 0.5)
	P.mutate(3, 0.1, 0.5)
	
	print('\nAfter Operation:')
	for I in P.individuals:
		print(I.chrom.tolist())
