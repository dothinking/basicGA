import numpy as np



class Individual():
	'''individual of population'''
	def __init__(self, dim, lbound, ubound):
		'''dimension of individual'''
		self._lbound = lbound
		self._ubound = ubound

		seeds = np.random.rand(dim)		
		self._value = lbound + (ubound-lbound)*seeds

	@property
	def value(self):
		return self._value

	def mutate(self, positions, rate):
		'''
		positions: mutate positions, list
		rate: mutate rate
		'''
		for pos in positions:
			if np.random.rand() < 0.5:
				self._value[pos] -= (self._value[pos]-self._lbound[pos])*rate
			else:
				self._value[pos] += (self._ubound[pos]-self._value[pos])*rate
	

class GA():
	'''collection of individuals'''
	def __init__(self, evaluation, dim, **kw):
		'''
		size: population size
		dim : dimension of individual
		evaluation: evaluation function
		'''
		self._evaluation = evaluation
		self._dim = dim

		# default parameters
		self._param = {
			'lbound': [float('-inf')] * dim,
			'ubound': [float('inf')] * dim,
			'size'	: 50,
			'max_generation': 100,
			'crossover_rate': 0.9,
			'mutation_rate'	: 0.1,
			'crossover_alpha': 0.5
		}		
		self._param.update(kw)

		# initialize population
		lbound = np.array(self._param['lbound'])
		ubound = np.array(self._param['ubound'])
		self._population = [Individual(dim, lbound, ubound) for i in range(size)]

	@property
	def evaluate(self):
		return np.array([self._evaluation(I.value) for I in self._population])	

	@property
	def fitness(self):
		'''
		we're trying to minimize the evaluation value,
		so the fitness is defined as: sum(abs(V))-Vi
		'''
		values = self.evaluate
		fitness = np.sum(np.abs(values)) - values
		return fitness/np.sum(fitness)

	def selection(self):
		'''select individuals by Roulette'''
		probabilities = np.cumsum(self.fitness)
		selected_population = [self._population[np.sum(np.random.rand()>=probabilities)] for i in range(self._size)]
		self._population = selected_population

	def crossover(self):
		random_population = np.random.permutation(self._population)

		for individual_a, individual_b in zip(self._population, random_population):

	def mutation(self, num=1, t=0):
		'''mutation in Generation t
		num: count of mutation position
		'''
		rate = 1.0 - np.random.rand()**(1.0-t/self._param['max_generation'])
		for individual in self._population:
			if np.random.rand() < self._param['mutation_rate']:
				individual.mutate(np.random.randint(0, self.dim, size=(num,)), rate)



if __name__ == '__main__':

	# f = lambda x : x-0.5

	# P = Population(4,1,f)

	

	# print(P._population)
	# P.selection()
	# print(P._population)

	# print(list(range(0,5,2)))

	x = np.random.uniform(0,10,size = (10,))

	# x = [1,2,3]

	print(np.random.permutation(x))