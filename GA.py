#----------------------------------------------------------
# Simple Genetic Algorithm
#----------------------------------------------------------
import numpy as np
import copy
from GAIndividual import Individual
from GAPopulation import Population


class GA():
	'''Simple Genetic Algorithm'''
	def __init__(self, fun_evaluation, dimension, lbound=None, ubound=None, **kw):
		'''
		dimension : dimension of individual
		fun_evaluation: evaluation function
		'''
		self._fun_evaluation = fun_evaluation
		self._dimension = dimension
		self._lbound = np.array(lbound) if lbound else [-1e6] * dimension
		self._ubound = np.array(ubound) if ubound else [1e6] * dimension

		# default parameters
		self._param = {
			'size'					: 32,
			'max_generation'		: 100,
			'fitness' 				: lambda x: np.sum(np.abs(x))-x,
			'selection_mode'		: 'roulette', 	# or 'elite'
			'selection_elite'		: True, 		# keep the best individual to next generation
			'crossover_rate'		: 0.9, 			# adaptive rate if list, e.g. [0.5, 0.9]
			'crossover_alpha'		: 0.0,			# factor to cross two float gene
			'mutation_rate'			: 0.08
		}
		self._param.update(kw)		

		# initialize population
		self._population = Population(fun_evaluation, self._param['fitness'], self._param['size'])

	def _initialize(self):
		'''initialize inidividuals of population'''
		self._population.initialize(self._dimension, self._lbound, self._ubound)

	def solve(self):
		'''
		solve the problem based on Simple GA process
		two improved methods could be considered:
			a) elitism mechanism: keep the best individual, i.e. skip the selection, crossover, mutation operations
			b) adaptive mechenism: adaptive crossover rate, adaptive mutation megnitude. this mechnism is considered
				automatically when self._param['crossover_rate'] is a list, e.g. [0.5,0.9]
		'''

		# initialize population
		self._initialize()

		# improve methods
		elitism = self._param['selection_elite']
		adaptive = isinstance(self._param['crossover_rate'], list)

		# solving process
		the_best = self._population.best if elitism else None
		for current_gen in range(1, self._param['max_generation']+1):

			# adaptive 1: fitness function for selection evaluation
			f = (lambda x: np.exp(x/0.99**(current_gen-1))) if adaptive else None
			self._population.select(self._param['selection_mode'], f)

			# adaptive 2: crossover rate
			self._population.crossover(self._param['crossover_rate'], self._param['crossover_alpha'])

			# adaptive 3: mutation rate
			rate = 1.0 - np.random.rand()**(1.0-current_gen/self._param['max_generation']) if adaptive else np.random.rand()
			self._population.mutate(self._dimension, self._param['mutation_rate'], rate)

			# update current population with the best individual ever
			if elitism:
				current_best = self._population.best
				if current_best.evaluation > the_best.evaluation:
					self._population.individuals[-1] = copy.deepcopy(the_best) # replace the last one by default
				else:
					the_best = copy.deepcopy(current_best)

		# return the best individual
		return self._population.best
		

if __name__ == '__main__':

	# schaffer-N4
	# sol: x=[0,1.25313], min=0.292579
	schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

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

	g = GA(schaffer_n4, 2, [-10,-10], [10,10], **kw)
	I = g.solve()

	x = [0,1.25313] 
	print('{0} : {1}'.format(I.evaluation, I.chrom))
	print('error: {:<3f} %'.format((I.evaluation/schaffer_n4(x)-1.0)*100))
