#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np
from .Individual import Individual


class DecimalFloatIndividual(Individual):
	'''
	dicimal encoded individual, the solutions are float elements
	ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
	'''
	
	def init_solution(self, ranges):
		'''
		initialize random solution in `ranges`
		ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
		'''		
		self._ranges = np.array(ranges)
		self._dimension = self._ranges.shape[0]

		# initialize solution within [lb, ub]
		seeds = np.random.random(self._dimension)
		lb = self._ranges[:, 0]
		ub = self._ranges[:, 1]
		self._solution = lb + (ub-lb)*seeds



class DecimalIntegerIndividual(Individual):
	'''
	dicimal encoded individual, the solutions are integer elements
	ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
	'''

	def init_solution(self, ranges):
		'''
		initialize random integer solution in `ranges`
		ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
		'''		
		self._ranges = np.array(ranges)
		self._dimension = self._ranges.shape[0]

		# initialize solution within [lb, ub]
		seeds = np.random.random(self._dimension)
		lb = self._ranges[:, 0]
		ub = self._ranges[:, 1]
		self._solution = np.rint(lb + (ub-lb)*seeds)


	@property
	def solution(self):
		return self._solution

	@solution.setter
	def solution(self, solution):
		self._solution = np.rint(solution)
