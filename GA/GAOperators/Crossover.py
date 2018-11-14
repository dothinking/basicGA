#----------------------------------------------------------
# GA Operator: crossover
#----------------------------------------------------------
import numpy as np
import copy
from .Operators import Crossover
from GA.GAPopulation.DecimalIndividual import DecimalFloatIndividual, DecimalIntegerIndividual
from GA.GAPopulation.SequenceIndividual import UniqueSeqIndividual, UniqueLoopIndividual, ZeroOneSeqIndividual

class DecimalCrossover(Crossover):
	'''
	crossover operation for Decimal encoded individuals:
	generate two new individuals by linear interpolation between two selected individuals
	'''
	def __init__(self, rate=0.8, alpha=0.5):
		'''
		crossover operation:
			- rate : propability of crossover. adaptive rate when it is a list, e.g. [0.6,0.9]
			- alpha: factor for crossing two chroms, [0,1]
		'''
		super().__init__(rate, alpha)
		self._individual_class = [DecimalFloatIndividual, DecimalIntegerIndividual]

	@staticmethod
	def cross_individuals(individual_a, individual_b, pos, alpha):
		'''
		generate two child individuals based on parent individuals:
			- pos  : 0-1 vector to specify positions for crossing
			- alpha: linear ratio to interpolate two genes, exchange two genes if alpha is 0.0
		'''

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
	

class SequencePMXCrossover(Crossover):
	'''
	Partially Mapped Crossover Operator(https://doi.org/10.1155/2017/7430125):
		- exchange genes at random positions
		- adjust to avoid duplicated genes
	'''
	def __init__(self, rate=0.8):
		'''
		crossover operation:
			- rate: propability of crossover. adaptive rate when it is a list, e.g. [0.6,0.9]
		'''
		super().__init__(rate)
		self._individual_class = [UniqueSeqIndividual, UniqueLoopIndividual]


	@staticmethod
	def cross_individuals(individual_a, individual_b, pos, alpha):
		'''
		Partially Mapped Crossover Operator(https://doi.org/10.1155/2017/7430125):
			- pos  : 0-1 vector to specify positions for crossing
			- alpha: not used
		'''
		solution_a = individual_a.solution.copy()
		solution_b = individual_b.solution.copy()

		# elements to be exchanged
		exchange_a, exchange_b = solution_a[pos], solution_b[pos]

		# unique elements among the exchanged elements
		diff_a = exchange_a[~np.isin(exchange_a, exchange_b)]
		diff_b = exchange_b[~np.isin(exchange_b, exchange_a)]

		# fix the duplicated elements
		solution_a[np.isin(solution_a, diff_b)] = diff_a
		solution_b[np.isin(solution_b, diff_a)] = diff_b

		# exchange specified elements finally
		solution_a[pos], solution_b[pos] = solution_b[pos], solution_a[pos]

		# return new individuals
		new_individual_a = individual_a.__class__(individual_a.ranges)
		new_individual_b = individual_b.__class__(individual_b.ranges)

		new_individual_a.solution = solution_a
		new_individual_b.solution = solution_b

		return new_individual_a, new_individual_b


class SequenceOXCrossover(Crossover):
	'''
	Order Crossover Operator(https://doi.org/10.1155/2017/7430125):
		- exchange genes at random positions
		- adjust to avoid duplicated genes
	'''
	def __init__(self, rate=0.8):
		'''
		crossover operation:
			- rate: propability of crossover. adaptive rate when it is a list, e.g. [0.6,0.9]
		'''
		super().__init__(rate)
		self._individual_class = [UniqueSeqIndividual, UniqueLoopIndividual]


	@staticmethod
	def cross_individuals(individual_a, individual_b, pos, alpha):
		'''
		Partially Mapped Crossover Operator(https://doi.org/10.1155/2017/7430125):
			- pos  : 0-1 vector to specify positions for crossing
			- alpha: not used
		'''
		solution_a = individual_a.solution
		solution_b = individual_b.solution

		# elements to be exchanged
		exchange_a, exchange_b = solution_a[pos], solution_b[pos]

		# unique elements
		unique_a = solution_a[~np.isin(solution_a, exchange_b)]
		unique_b = solution_b[~np.isin(solution_b, exchange_a)]

		# return new individuals
		new_individual_a = individual_a.__class__(individual_a.ranges)
		new_individual_b = individual_b.__class__(individual_b.ranges)

		new_individual_a.solution = np.concatenate((exchange_b, unique_a))
		new_individual_b.solution = np.concatenate((exchange_a, unique_b))

		return new_individual_a, new_individual_b