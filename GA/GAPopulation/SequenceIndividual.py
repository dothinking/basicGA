#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np
from .Individual import Individual


class UniqueSeqIndividual(Individual):
	'''
	sequence encoded individual: unique numbers in a certain order	
	'''

	def __init__(self, ranges, loop=False):
		'''
		ranges: int, e.g. ranges=5 -> 0,1,2,3,4
		loop: bool, if True: 0,1,2,3,4 = 1,2,3,4,0 = 3,4,0,1,2 = ...
		'''
		self.loop = loop
		super().__init__(ranges)
	
	def init_solution(self, ranges):
		'''
		initialize random solution: e.g. 0,3,2,1,4
		'''	
		if not isinstance(ranges, int) or ranges<=1:
			raise ValueError('the sequence range should be larger than 1')

		self._ranges = ranges
		self._dimension = ranges
		seq = np.random.choice(ranges, ranges, replace=False)
		self._solution = self.unique_sequence(seq)
		print(self._solution)

	@property
	def solution(self):
		return self._solution

	@solution.setter
	def solution(self, solution):
		self._solution = self.unique_sequence(solution)

	def unique_sequence(self, sequence):
		'''
		only relative order is considered for a sequece loop, 
		so represent the loop from element 0
		e.g. 1,4,0,2,3 -> 0,2,3,1,4
		'''

		if not self.loop: # do nothing if the sequence is not a loop
			return sequence

		# find position of element 0
		arg = np.argwhere(sequence==0) # [[3]]
		pos = arg[0][0]

		# exchange two sections seperated by element 0
		unique_seq = np.empty_like(sequence)
		unique_seq[0:-pos], unique_seq[-pos:] = sequence[pos:], sequence[0:pos]
		
		return unique_seq



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
