#----------------------------------------------------------
# GA Operator: mutation
#----------------------------------------------------------
import numpy as np
from .Operators import Mutation
from GA.GAPopulation.DecimalIndividual import DecimalFloatIndividual, DecimalIntegerIndividual
from GA.GAPopulation.SequenceIndividual import UniqueSeqIndividual, UniqueLoopIndividual, ZeroOneSeqIndividual

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
		super().__init__(rate)

		# this operator is only available for DecimalIndividual
		self._individual_class = [DecimalFloatIndividual, DecimalIntegerIndividual]

	@staticmethod
	def mutate_individual(individual, positions, alpha, fun_evaluation=None):
		'''
		mutation method for decimal encoded individual:
		to add a random deviation for gene in specified positions
		- positions: 0-1 vector to specify positions for crossing
		- alpha: mutatation magnitude
		'''

		# for a gene G in range [L, U],
		# option 0: G = G + (U-G)*alpha
		# option 1:	G = G + (L-G)*alpha	

		# mutation options:
		p = np.random.choice(2,individual.dimension)

		# lower/upper bound
		L, U = individual.ranges[:,0], individual.ranges[:,1]
		
		# combine two mutation method
		diff = ((U-individual.solution)-p*(U-L))*positions*alpha
		solution = individual.solution + diff

		return solution			
		


class UniqueSeqMutation(Mutation):
	'''
	mutation operation for unique sequence individuals:
	exchange genes at random positions
	'''
	def __init__(self, rate):
		'''
		mutation operation:
		rate: propability of mutation, [0,1]
		'''
		super().__init__(rate)

		# this operator is only available for UniqueSeqIndividual
		self._individual_class = [UniqueSeqIndividual, UniqueLoopIndividual]


	@staticmethod
	def _mutate_positions(dimension):
		'''select random and continuous positions'''
		# start, end position
		pos = np.random.choice(dimension, 2, replace=False)
		start, end = pos.min(), pos.max()
		positions = np.zeros(dimension).astype(np.bool)
		positions[start:end+1] = True
		return positions

	@staticmethod
	def mutate_individual(individual, positions, alpha):
		'''
		reverse genes at specified positions:
		- positions: 0-1 vector to specify positions
		- alpha: probability to accept a worse solution
		'''
		solution = individual.solution.copy()		
		solution[positions] = solution[positions][::-1] # reverse genes at specified positions
		return solution