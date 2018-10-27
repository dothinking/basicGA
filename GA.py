#----------------------------------------------------------
# Simple Genetic Algorithm
#----------------------------------------------------------
import numpy as np
from GAIndividual import Individual
from GAPopulation import Population

class GA():
	'''Simple Genetic Algorithm'''
	def __init__(self, fun_evaluation, dimention, **kw):
		'''
		dimention : dimension of individual
		fun_evaluation: evaluation function
		'''
		self._fun_evaluation = fun_evaluation

		# default parameters
		self._param = {
			'var_dim' 				: dimention,
			'lbound'				: [-1e9] * dimention,
			'ubound'				: [1e9] * dimention,
			'size'					: 32,
			'max_generation'		: 100,
			'fitness' 				: lambda x: np.sum(np.abs(x))-x,
			'selection_mode'		: 'roulette',
			'crossover_rate'		: 0.9,			
			'crossover_alpha'		: 0.0,
			'crossover_rate_range'	: [0.5, 0.9],
			'mutation_rate'			: 0.08
		}
		self._param.update(kw)		

		# initialize population
		self._population = Population(fun_evaluation, self._param['fitness'], self._param['size'])

	def _initialize(self):
		'''initialize inidividuals of population'''
		lbound = np.array(self._param['lbound'])
		ubound = np.array(self._param['ubound'])
		self._population.initialize(self._param['var_dim'], lbound, ubound)

	def _SGA_solve(self):
		'''	Simple Genetic Algorithm - the selection mode is roulette by default
		'''
		for current_gen in range(1, self._param['max_generation']+1):
			# GA operations
			self._population.select()
			self._population.crossover(self._param['crossover_rate'], self._param['crossover_alpha'])
			self._population.mutate(self._param['var_dim'], self._param['mutation_rate'], np.random.rand())

	def _EGA_solve(self):
		'''Elite controll: inherit the best individual from parent population
		'''
		history_best = self._population.best
		current_best = history_best

		for current_gen in range(1, self._param['max_generation']+1):

			if current_best.evaluation > history_best.evaluation:
				self._population.individuals[-1] = history_best
			else:
				history_best = current_best

			# GA operations
			self._population.select(method='elite')
			self._population.crossover(self._param['crossover_rate'], self._param['crossover_alpha'])
			self._population.mutate(self._param['var_dim'], self._param['mutation_rate'],  np.random.rand())

	def _AGA_solve(self):
		'''Adaptive Genetic Algorithm
		'''		
		for current_gen in range(1, self._param['max_generation']+1):

			# adaptive 1: fitness function for selection evaluation
			f = lambda x: np.exp(x/0.99**(current_gen-1))
			self._population.select(self._param['selection_mode'], f)

			# adaptive 2: crossover rate
			self._population.crossover(self._param['crossover_rate_range'], self._param['crossover_alpha'])

			# adaptive 3: mutation rate
			rate = 1.0 - np.random.rand()**(1.0-current_gen/self._param['max_generation'])
			self._population.mutate(self._param['var_dim'], self._param['mutation_rate'], rate)


	def solve(self, solver='SGA'):
		'''solve based on specified solver'''
		# initialize population
		self._initialize()

		# solve
		methods = {
			'SGA': self._SGA_solve,
			'EGA': self._EGA_solve,
			'AGA': self._AGA_solve,
		}
		methods.get(solver, self._SGA_solve)()

		# return the best individual
		return self._population.best
		

if __name__ == '__main__':

	# schaffer-N4
	# sol: x=[0,1.25313], min=0.292579
	schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

	kw = {
		'lbound': [-10, -10],
		'ubound': [10, 10],
		'size'	: 100,
		'max_generation': 50,
		# 'fitness': lambda x: np.exp(-x),
		'selection_mode': 'elite',
		'crossover_rate': 0.8,		
		'crossover_alpha': 0.1,
		'mutation_rate'	: 0.05
	}

	g = GA(schaffer_n4, 2, **kw)	
	# I = g.solve()
	# print('Best individual: {0}'.format(I.chrom))
	# print('Output: {0}'.format(I.evaluation))

	# I = g.solve('SGA')
	# print('Best individual: {0}'.format(I.chrom))
	# print('Output: {0}'.format(I.evaluation))

	# I = g.solve('EGA')
	# print('Best individual: {0}'.format(I.chrom))
	# print('Output: {0}'.format(I.evaluation))

	x = [g.solve('SGA').evaluation/0.292579-1 for i in range(10)]
	y = [g.solve('EGA').evaluation/0.292579-1 for i in range(10)]
	z = [g.solve('AGA').evaluation/0.292579-1 for i in range(10)]

	print(sum(x)/10)
	print(sum(y)/10)
	print(sum(z)/10)
