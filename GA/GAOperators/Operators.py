#----------------------------------------------------------
# GA Operator Base Class: selection, crossover, mutation
#----------------------------------------------------------
import numpy as np
import copy

# SELECTION
class Selection:
	def select(self, population):
		'''
		- population: where the individuals from
		- return: the selected individuals
		'''
		raise NotImplementedError


# CROSSOVER
class Crossover:
	'''
	this operation is only available for Individual class defined in self._individual_class
	'''
	def __init__(self, rate=0.8, alpha=0.0):
		'''
		crossover operation:
		- rate: propability of crossover. adaptive rate when it is a list, e.g. [0.6,0.9]
		'''
		if isinstance(rate, float) and 0.0<rate<=1.0:
			pass
		elif isinstance(rate, (list, tuple)) and len(rate)==2 and rate[1]>=rate[0]:
			pass
		else:
			raise ValueError('crossover rate should be a float in range [0,1] or a list with two element')

		self._rate = rate
		self._alpha = alpha
		self._individual_class = None

	def _adaptive_rate(self, individual_a, individual_b, population):
		'''
		get the adaptive rate when cross over two individuals:
		if f<f_avg  then rate = range_max,
		if f>=f_avg then rate = range_max-(range_max-range_min)*(f-f_avg)/(f_max-f_avg),
		where f=max(individual_a, individual_b)
		'''
		if not isinstance(self._rate, (list, tuple)):
			return self._rate

		fitness = [I.fitness for I in population.individuals]
		fit_max, fit_avg = np.max(fitness), np.mean(fitness)
		fit = max(individual_a.fitness, individual_b.fitness)
		if fit_max-fit_avg:
			return self._rate[1] if fit<fit_avg else self._rate[1] - (self._rate[1]-self._rate[0])*(fit-fit_avg)/(fit_max-fit_avg)
		else:
			return (self._rate[0]+self._rate[1])/2.0

	@property
	def individual_class(self):
		return self._individual_class

	@staticmethod
	def cross_individuals(individual_a, individual_b, pos, alpha):
		'''
        generate two individuals based on parent individuals:
            - individual_a, individual_b: the selected individuals
            - pos  : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: two generated individuals
        '''
		raise NotImplementedError

	@staticmethod
	def _cross_positions(dimension):
		'''generate a random and continuous range of positions for crossover'''
		# start, end position
		pos = np.random.choice(dimension, 2)
		start, end = pos.min(), pos.max()
		positions = np.zeros(dimension).astype(np.bool)
		positions[start:end+1] = True
		return positions
	
	def cross(self, population):
		'''
		population: population to be crossed. population should be evaluated in advance 
					since the crossover may be based on individual fitness.
		'''
		
		random_population = np.random.permutation(population.individuals) # random order
		new_individuals, count = [], 0
		for individual_a, individual_b in zip(population.individuals, random_population):
			# crossover
			if np.random.rand() <= self._adaptive_rate(individual_a, individual_b, population):
				# random position to cross
				pos = self._cross_positions(individual_a.dimension)
				child_individuals = self.cross_individuals(individual_a, individual_b, pos, self._alpha)
				new_individuals.extend(child_individuals)

			# skip crossover, but copy parents directly
			else:
				new_individuals.append(copy.deepcopy(individual_a))
				new_individuals.append(copy.deepcopy(individual_b))

			# generate two child at one crossover
			count += 2

			# stop when reach the population size
			if count > population.size:
				break

		# the count of new individuals may lower than the population size
		# since same parent individuals for crossover would be ignored
		# so when count < size, param `replace` for choice() is True,
		# which means dupilcated individuals are necessary
		return np.random.choice(new_individuals, population.size, replace=False)

# MUTATION
class Mutation:
	'''
	this operation is only available for Individual class defined in self._individual_class
	'''
	def __init__(self, rate):
		self._rate = rate
		self._individual_class = None

	@property
	def individual_class(self):
		return self._individual_class	

	@staticmethod
	def mutate_individual(individual, positions, alpha):
		'''
        get mutated solution based on the selected individual:
            - individual: the selected individual
            - positions : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: the mutated solution
        '''
		raise NotImplementedError

	@staticmethod
	def _mutate_positions(dimension):
		'''select num positions from dimension to mutate'''
		num = np.random.randint(dimension)+1
		pos = np.random.choice(dimension, num, replace=False)
		positions = np.zeros(dimension).astype(np.bool)
		positions[pos] = True
		return positions
	
	def mutate(self, population, alpha=None):
		'''
		- population: population to be selected. 		
		- alpha: additional params
		'''
		for individual in population.individuals:
			if np.random.rand() > self._rate: continue
			pos = self._mutate_positions(individual.dimension)
			individual.solution = self.mutate_individual(individual, pos, alpha)			
			individual.init_evaluation() # reset evaluation
