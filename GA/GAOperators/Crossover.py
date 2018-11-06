#----------------------------------------------------------
# GA Operator: crossover
#----------------------------------------------------------
import numpy as np
from .Operators import Crossover


class DecimalCrossover(Crossover):
	'''
	crossover operation for Decimal encoded individuals:
	linear interpolate the selected two individuals to generate two new ones
	'''
	def __init__(self, rate=0.8, alpha=0.5):
		'''
		crossover operation:
			rate: propability of crossover. adaptive rate when it is a list, e.g. [0.6,0.9]
					if f<f_avg then rate = range_max
					if f>=f_avg then rate = range_max-(range_max-range_min)*(f-f_avg)/(f_max-f_avg)
					where f=max(individual_a, individual_b)
			alpha: factor for crossing two chroms, [0,1]
		'''
		# parameters check is skipped
		self.rate = rate
		self.alpha = alpha

	@staticmethod
	def cross_individuals(individual_a, individual_b, alpha):
		'''
		generate two child individuals based on parent individuals:
		new values are calculated at random positions
		alpha: linear ratio to cross two genes, exchange two genes if alpha is 0.0
		'''
		# random positions to be crossed
		pos = np.random.rand(individual_a.dimension) <= 0.5

		# cross value
		temp = (individual_b.solution-individual_a.solution)*pos*(1-alpha)
		new_value_a = individual_a.solution + temp
		new_value_b = individual_b.solution - temp

		# return new individuals
		new_individual_a = individual_a.__class__(individual_a.ranges)
		new_individual_b = individual_b.__class__(individual_b.ranges)

		new_individual_a.solution = new_value_a
		new_individual_b.solution = new_value_b

		return new_individual_a, new_individual_b

	def cross(self, population):
		adaptive = isinstance(self.rate, list)
		# adaptive rate
		if adaptive:
			fitness = [I.fitness for I in population.individuals]
			fit_max, fit_avg = np.max(fitness), np.mean(fitness)

		new_individuals = []		
		random_population = np.random.permutation(population.individuals) # random order
		num = int(population.size/2.0)+1

		for individual_a, individual_b in zip(population.individuals[0:num+1], random_population[0:num+1]):			
			# adaptive rate
			if adaptive:
				fit = max(individual_a.fitness, individual_b.fitness)
				if fit_max-fit_avg:
					i_rate = self.rate[1] if fit<fit_avg else self.rate[1] - (self.rate[1]-self.rate[0])*(fit-fit_avg)/(fit_max-fit_avg)
				else:
					i_rate = (self.rate[0]+self.rate[1])/2.0
			else:
				i_rate = self.rate

			# crossover
			if np.random.rand() <= i_rate:
				child_individuals = self.cross_individuals(individual_a, individual_b, self.alpha)
				new_individuals.extend(child_individuals)
			else:
				new_individuals.append(individual_a)
				new_individuals.append(individual_b)

		population.individuals = np.array(new_individuals[0:population.size])
