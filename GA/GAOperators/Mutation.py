#----------------------------------------------------------
# GA Operator: mutation
#----------------------------------------------------------
import numpy as np
import copy
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
	def mutate_individual(individual, positions, alpha):
		'''
		mutation method for decimal encoded individual:
		to add a random deviation for gene in specified positions
		- positions: 0-1 vector to specify positions for crossing
		- alpha: mutatation magnitude
		'''

		# for a gene G in range [L, U], either:
		# 	G = G - (G-L)*alpha
		# or:
		#   G = G + (U-G)*alpha

		# mutation options:
		p = np.random.rand(individual.dimension)<=0.5

		# lower/upper bound
		L, U = individual.ranges[:,0], individual.ranges[:,1]
		
		# combine two mutation method
		diff = ((U-individual.solution)-p*(U-L))*positions*alpha
		sol = individual.solution + diff	
				
		# set new solution and reset evaluation
		individual.solution = sol
		individual.init_evaluation()


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
	def mutate_individual(individual, positions, alpha):
		'''
		exchange genes at specified positions:
		- positions: 0-1 vector to specify positions for crossing
		- alpha: additional param, ignored
		'''

		# reorder genes at specified positions
		genes = np.random.permutation(individual.solution[positions])		
				
		# set new solution and reset evaluation
		individual.solution[positions] = genes
		individual.init_evaluation()
