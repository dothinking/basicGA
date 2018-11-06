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
	def cross(self, population):
		'''
		population: population to be crossed. population should be evaluated in advance 
					since the crossover may be based on individual fitness.
		'''
		raise NotImplementedError


# MUTATION
class Mutation:	
	def mutate(self, population, alpha=None):
		'''
		- population: population to be selected. 
		- alpha		: additional params
		'''
		raise NotImplementedError
