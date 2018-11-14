#----------------------------------------------------------
# Population class
#----------------------------------------------------------
import numpy as np


class Population:
	'''collection of individuals'''
	def __init__(self, individual, size=50):
		'''
		individual   : individual template
		size         : count of individuals		
		'''
		self.individual = individual
		self.size = size
		self.individuals = None


	def initialize(self):
		'''initialization for next generation'''
		IndvClass = self.individual.__class__
		self.individuals = np.array([IndvClass(self.individual.ranges) for i in range(self.size)], dtype=IndvClass)

	@property
	def best(self):
		'''get best individual according to evaluation value'''
		# collect evaluations
		evaluation = np.array([I.evaluation for I in self.individuals])

		# get the minimum position
		pos = np.argmin(evaluation)
		return self.individuals[pos]

	def evaluate(self, fun_evaluation, fun_fitness):
		'''
		calculate objectibe value and fitness for each individual.
			- fun_evaluation: objective function
			- fun_fitness  	: population fitness based on evaluation
		'''

		# get the value directly if it has been calculated before
		evaluation = np.array([fun_evaluation(I.solution) if I.evaluation is None else I.evaluation for I in self.individuals])		

		# calculate fitness
		fitness = fun_fitness(evaluation)
		fitness = fitness/fitness.sum() # normalize
		
		# set attributes for each individual
		for I, e, f in zip(self.individuals, evaluation, fitness):
			I.evaluation = e
			I.fitness = f