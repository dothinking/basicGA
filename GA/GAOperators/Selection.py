#----------------------------------------------------------
# GA Operator: selection
#----------------------------------------------------------
import numpy as np
import copy
from .Operators import Selection

class RouletteWheelSelection(Selection):
	'''
	select individuals by Roulette Wheel:
	individuals are selected by a probability on its fitness
	'''	
	def select(self, population):
		fitness = np.array([I.fitness for I in population.individuals])
		selected_individuals = np.random.choice(population.individuals, population.size, p=fitness)

		# pay attention to deep copy these objects		
		return np.array([copy.deepcopy(I) for I in selected_individuals])


class LinearRankingSelection(Selection):
	'''
	select individuals by Roulette Wheel:
	individuals are selected by a probaility on its ranking
	'''
	def __init__(self, rate=100):
		'''
		rate: probability ratio of the best individual to the worst
			it shows the relative probability of the best/worst individual to be selected
		'''
		if rate<1.0:
			raise ValueError('the selection probability of the best individual should be larger than the worst') 
			
		self.rate = rate

	def select(self, population):
		fitness = np.array([I.fitness for I in population.individuals])
		pos = np.argsort(fitness)
		rank_fitness = 1.0 + (self.rate-1.0)/(population.size-1)*np.arange(population.size)
		# normalize
		rank_fitness = rank_fitness/(population.size*(1+self.rate)/2.0) # np.sum(rank_fitness) = population.size*(1+self.rate)/2
		selected_individuals = np.random.choice(population.individuals[pos], population.size, p=rank_fitness)
		
		return np.array([copy.deepcopy(I) for I in selected_individuals])

