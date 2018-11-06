#----------------------------------------------------------
# GA Operator Base Class: selection, crossover, mutation
#----------------------------------------------------------


# SELECTION
class Selection:
	def select(self, population):
		'''
		population: population to be selected. population should be evaluated in advance 
					since the selection is based on individual fitness.
		'''
		raise NotImplementedError


# CROSSOVER
class Crossover:
	'''
	this operation is only available for Individual class defined in self._individual_class
	'''
	def __init__(self):
		self._individual_class = None

	@property
	def individual_class(self):
		return self._individual_class
	
	def cross(self, population):
		'''
		population: population to be crossed. population should be evaluated in advance 
					since the crossover may be based on individual fitness.
		'''
		raise NotImplementedError


# MUTATION
class Mutation:
	'''
	this operation is only available for Individual class defined in self._individual_class
	'''
	def __init__(self):
		self._individual_class = None

	@property
	def individual_class(self):
		return self._individual_class
	
	def mutate(self, population, alpha=None):
		'''
		- population: population to be selected. 
		- alpha		: additional params
		'''
		raise NotImplementedError
