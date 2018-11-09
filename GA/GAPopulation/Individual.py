#----------------------------------------------------------
# Individual Base Class:
# - properties:
# 	- _ranges
# 	- _dimension
# 	- _solution
# 	- evaluation
# 	- fitness
# - methods to be implemented
# 	- init_solution(ranges): initialize random solution
#   - solution
#----------------------------------------------------------

class Individual:
	'''base class: individual of population'''
	def __init__(self, ranges):
		'''
		ranges: element ranges of solution
		'''
		# random solution
		self.init_solution(ranges)

		# evaluation and fitness
		self.init_evaluation()

	def init_solution(self, ranges):
		'''
		initialize a random solution, three required properties:
        - self._ranges: ranges for the solution
        - self._dimension: count of variables
        - self._solution: solution for the problem
        '''
		self._ranges = ranges
		self._dimension = None
		self._solution = None

	def init_evaluation(self):
		self.evaluation = None
		self.fitness = None

	@property
	def ranges(self):
		return self._ranges

	@property
	def dimension(self):
		return self._dimension

	@property
	def solution(self):
		return self._solution

	@solution.setter
	def solution(self, solution):
		self._solution = solution
	
