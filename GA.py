import numpy as np

class Individual():
	'''individual of population'''
	def __init__(self, dim, lbound, ubound):
		'''dimension of individual'''
		self._dim = dim
		self._lbound = lbound
		self._ubound = ubound
		self._chrom = np.empty((dim,))

	def initialize(self):
		'''initialize random values in [lbound, ubound]'''
		seeds = np.random.random(self._dim)		
		self._chrom = self.lbound + (self.ubound-self.lbound)*seeds

	@property
	def dimension(self):
		return self._dim

	@property
	def lbound(self):
		return self._lbound

	@property
	def ubound(self):
		return self._ubound

	@property
	def chrom(self):
		return self._chrom

	@chrom.setter
	def chrom(self, chrom):
		assert self.dimension == chrom.shape[0]
		assert (chrom>=self._lbound).all() and (chrom<=self._ubound).all()
		self._chrom = chrom	

	def mutate(self, positions, alpha):
		'''
		positions: mutating gene positions, list
		alpha: mutatation magnitude
		'''
		for pos in positions:
			if np.random.rand() < 0.5:
				self._chrom[pos] -= (self._chrom[pos]-self._lbound[pos])*alpha
			else:
				self._chrom[pos] += (self._ubound[pos]-self._chrom[pos])*alpha
	

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
		self._evaluation = None

	def initialize(self, dim, lbound, ubound):
		'''initialization for next generation'''
		self._individuals = np.array([Individual(dim, lbound, ubound) for i in range(self._size)])
		for individual in self._individuals:
			individual.initialize()	

	@property
	def individuals(self):
		return self._individuals	

	@property
	def evaluation(self):
		if self._evaluation is None:
			self._evaluation = np.array([self._fun_evaluation(I.chrom) for I in self._individuals])
		return self._evaluation

	@property
	def fitness(self):
		'''
		we're trying to minimize the evaluation value,
		so the fitness is defined as: sum(abs(V))-Vi
		'''
		# fitness = np.sum(np.abs(self.evaluation)) - self.evaluation
		fitness = self._fun_fitness(self.evaluation)
		return fitness/np.sum(fitness)

	@property
	def best(self):
		pos = np.argmin(self.evaluation)
		return self._individuals[pos], self.evaluation[pos]

	def convergent(self, tol=1e-6):
		'''max deviation of all individuals'''
		max_eval, min_eval = np.max(self.evaluation), np.min(self.evaluation)
		return max_eval-min_eval <= tol	

	def _select_by_roulette(self):
		'''select individuals by Roulette'''
		return np.random.choice(self._individuals, self._size, p=self.fitness)

	def _select_by_elite(self):
		pos = np.argsort(self.fitness)
		elite_pos = pos[int(self._size/2):]		
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
		new_individual_a = Individual(individual_a.dimension, individual_a.lbound, individual_a.ubound)
		new_individual_b = Individual(individual_b.dimension, individual_b.lbound, individual_b.ubound)

		new_individual_a.chrom = new_value_a
		new_individual_b.chrom = new_value_b

		return new_individual_a, new_individual_b

	def select(self, sel_type='roulette'):
		'''select individuals'''

		methods = {
			'roulette': self._select_by_roulette,
			'elite'  : self._select_by_elite
		}
		select_method = methods.get(sel_type, self._select_by_roulette)
		
		self._individuals = select_method()
		self._evaluation = None # reset evaluation

	def crossover(self, rate, alpha):
		'''crossover operation'''
		new_individuals = []		
		random_population = np.random.permutation(self._individuals) # random order
		for individual_a, individual_b in zip(self._individuals, random_population):
			if np.random.rand() <= rate:
				child_individuals = self._cross_individuals(individual_a, individual_b, alpha)
				new_individuals.extend(child_individuals)
			else:
				new_individuals.append(individual_a)
				new_individuals.append(individual_b)

		self._individuals = np.random.choice(new_individuals, self._size, replace=False)
		self._evaluation = None # reset evaluation

	def mutate(self, num, rate, alpha):
		'''
		mutation operation:
		num: count of mutating gene
		rate: propability of mutation
		alpha: mutating magnitude
		'''
		self._evaluation = None # reset evaluation
		for individual in self._individuals:
			if np.random.rand() <= rate:
				pos = np.random.choice(individual.dimension, num, replace=False)
				individual.mutate(pos, alpha)


class GA():
	'''collection of individuals'''
	def __init__(self, fun_evaluation, dimention, **kw):
		'''
		dimention : dimension of individual
		fun_evaluation: evaluation function
		'''
		self._fun_evaluation = fun_evaluation

		# default parameters
		self._param = {
			'var_dim': dimention,
			'lbound': [-1e9] * dimention,
			'ubound': [1e9] * dimention,
			'size'	: 200,
			'max_generation': 10,
			'fitness' 		: lambda x: np.sum(np.abs(x))-x,
			'selection_mode': 'roulette',
			'crossover_rate': 0.9,			
			'crossover_alpha': 0.0,
			'mutation_rate'	: 0.08
		}
		self._param.update(kw)		

		# initialize population
		lbound = np.array(self._param['lbound'])
		ubound = np.array(self._param['ubound'])
		self._population = Population(fun_evaluation, self._param['fitness'], self._param['size'])
		self._population.initialize(dimention, lbound, ubound)


	def solve(self):

		for current_gen in range(1, self._param['max_generation']+1):
			# GA operations
			rate = 1.0 - np.random.rand()**(1.0-current_gen/self._param['max_generation'])
			self._population.select(self._param['selection_mode'])
			self._population.crossover(self._param['crossover_rate'], self._param['crossover_alpha'])
			self._population.mutate(np.random.randint(self._param['var_dim'])+1, self._param['mutation_rate'], rate)

			# evaluation
			_, val = self._population.best
			print('Generation {0}: {1}'.format(current_gen, val))

			if self._population.convergent():
				break

		# results
		individual, val = self._population.best
		print('Current generation: {0}'.format(current_gen))
		print('Best individual: {0}'.format(individual.chrom))
		print('Output: {0}'.format(val))

		return individual, val



if __name__ == '__main__':

	import math

	# schaffer-N4
	# sol: x=[0,1.25313], min=0.292579
	schaffer_n4 = lambda x: 0.5 + (math.cos(math.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

	kw = {
		'lbound': [-100, -100],
		'ubound': [100, 100],
		'size'	: 195,
		'max_generation': 30,
		'fitness': lambda x: np.exp(-x),
		'selection_mode': 'elite',
		'crossover_rate': 0.85,		
		'crossover_alpha': 0.75,
		'mutation_rate'	: 0.02
	}

	g = GA(schaffer_n4, 2, **kw)
	g.solve()
