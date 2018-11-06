#----------------------------------------------------------
# GA Operator: mutation
#----------------------------------------------------------
import numpy as np
import copy
from .Operators import Mutation
from GA.GAPopulation.DecimalIndividual import DecimalFloatIndividual, DecimalIntegerIndividual
from GA.GAPopulation.SequenceIndividual import UniqueSeqIndividual, ZeroOneSeqIndividual

class DecimalMutation(Mutation):
	'''
	mutation operation for decimal encoded individuals:
	add random deviations(positive/negtive) at random positions of the selected individual
	'''
	def __init__(self, rate):
		'''
		mutation operation:
		rate: propability of mutation, [0,1]
		'''
		self.rate = rate

		# this operator is only available for DecimalIndividual
		self._individual_class = [DecimalFloatIndividual, DecimalIntegerIndividual]

	def mutate_individual(self, individual, positions, alpha):
		'''
		positions: mutating gene positions, list
		alpha: mutatation magnitude
		'''
		# for pos in positions:
		# 	if np.random.rand() < 0.5:
		# 		individual.solution[pos] -= (individual.solution[pos]-individual.ranges[:,0][pos])*alpha
		# 	else:
		# 		individual.solution[pos] += (individual.ranges[:,1][pos]-individual.solution[pos])*alpha

		sol = copy.deepcopy(individual.solution)
		p = np.random.rand(positions.shape[0])<=0.5 # change to lower bound or upper bound
		L, U = individual.ranges[:,0][positions], individual.ranges[:,1][positions]
		sol[positions] += ((U-sol[positions])-p*(U-L))*alpha
		individual.solution = sol
				
		# reset evaluation
		individual.init_evaluation()

	
	def mutate(self, population, alpha):
		'''
		alpha: mutating magnitude
		'''
		for individual in population.individuals:
			if np.random.rand() > self.rate:
				continue

			# select random positions to mutate
			num = np.random.randint(individual.dimension) + 1
			pos = np.random.choice(individual.dimension, num, replace=False)
			self.mutate_individual(individual, pos, alpha)
