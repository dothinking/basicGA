#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np
from .Individual import Individual


class UniqueSeqIndividual(Individual):
	'''
	sequence encoded individual: unique numbers in a certain order
	ranges: int, e.g. ranges=5 -> 1,2,3,4,5
	'''
	
	def init_solution(self, ranges):
		'''
		initialize random solution: e.g. 5,3,2,1,4
		'''	
		if not isinstance(ranges, int) or ranges<=1:
			raise ValueError('the sequence range should be larger than 1')

		self._ranges = ranges
		self._dimension = ranges
		self._solution = np.random.choice(ranges, ranges, replace=False) + 1


class ZeroOneSeqIndividual(Individual):
	'''
	sequence encoded individual: 0-1 sequence
	ranges: int, length of 0-1 sequence
	'''

	def init_solution(self, ranges):
		'''
		initialize random 0-1 sequence: e.g. 1,0,1,1,0
		'''	
		if not isinstance(ranges, int) or ranges<=1:
			raise ValueError('the sequence range should be larger 1')

		self._ranges = ranges
		self._dimension = ranges
		self._solution = np.random.choice(2, ranges)