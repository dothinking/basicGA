#----------------------------------------------------------
# Simple Genetic Algorithm
#----------------------------------------------------------
import numpy as np
import copy


class GA():
	'''Simple Genetic Algorithm'''
	def __init__(self, population, selection, crossover, mutation, fun_fitness=lambda x:np.arctan(-x)+np.pi):
		'''
		fun_fitness: fitness based on objective values. minimize the objective by default
		'''
		self.population = population
		self.selection = selection
		self.crossover = crossover
		self.mutation = mutation
		self.fun_fitness = fun_fitness

	def run(self, fun_evaluation, gen=50, elitism=True):
		'''
		solve the problem based on Simple GA process
		two improved methods could be considered:
			a) elitism mechanism: keep the best individual, i.e. skip the selection, crossover, mutation operations
			b) adaptive mechenism: adaptive crossover rate, adaptive mutation megnitude. 
		'''

		# initialize population
		self.population.initialize()

		# solving process
		for n in range(1, gen+1):

			# the best individual in previous generation
			if elitism:
				the_best = copy.deepcopy(self.population.best(fun_evaluation, self.fun_fitness))

			# selection
			fitness, _ = self.population.fitness(fun_evaluation, self.fun_fitness)
			self.selection.select(self.population, fitness)

			# crossover
			self.crossover.cross(self.population)

			# mutation
			rate = 1.0 - np.random.rand()**((1.0-n/gen)**3)
			self.mutation.mutate(self.population, rate)

			# elitism mechanism: 
			# set a random individual as the best in previous generation
			if elitism:
				pos = np.random.randint(self.population.size)
				self.population.individuals[pos] = the_best

		# return the best individual
		return self.population.best(fun_evaluation, self.fun_fitness)
		

if __name__ == '__main__':

	from GAComponents import Individual, Population
	from GAOperators import RouletteWheelSelection, LinearRankingSelection, Crossover, Mutation

	# schaffer-N4
	# sol: x=[0,1.25313], min=0.292579
	schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

	ranges = [(-10, 10)] * 2

	I = Individual(ranges)
	P = Population(I, 50)
	S = RouletteWheelSelection()
	SL = LinearRankingSelection()
	C = Crossover([0.5, 0.9], 0.5)
	M = Mutation(0.12)

	g = GA(P, S, C, M)
	res = g.run(schaffer_n4, 200)	

	x = [0,1.25313] 
	print('{0} : {1}'.format(res.evaluation, res.solution))
	print('error: {:<3f} %'.format((res.evaluation/schaffer_n4(x)-1.0)*100))