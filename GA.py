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

	def run(self, fun_evaluation, gen=50, elitism=True, adaptive=True):
		'''
		solve the problem based on Simple GA process
		two improved methods could be considered:
			a) elitism mechanism: keep the best individual, i.e. skip the selection, crossover, mutation operations
			b) adaptive mechenism: adaptive crossover rate, adaptive mutation megnitude. 
		'''

		# initialize population
		self.population.initialize()

		# solving process
		the_best = self.population.best(fun_evaluation, self.fun_fitness) if elitism else None
		for n in range(1, gen+1):

			# adaptive 1: fitness function for selection evaluation
			m = n if n<100 else 100
			f = (lambda x: np.exp(x/0.99**m)) if adaptive else None
			fitness, _ = self.population.fitness(fun_evaluation, self.fun_fitness, f)
			self.selection.select(self.population, fitness)

			# adaptive 2: crossover rate
			self.crossover.cross(self.population)

			# adaptive 3: mutation rate
			rate = 1.0 - np.random.rand()**(1.0-n/gen) if adaptive else np.random.rand()
			self.mutation.mutate(self.population, rate)

			# update current population with the best individual ever
			if not elitism: continue
			current_best = self.population.best(fun_evaluation, self.fun_fitness)
			if current_best.evaluation > the_best.evaluation:
				self.population.individuals[-1] = copy.deepcopy(the_best) # replace the last one by default
			else:
				the_best = copy.deepcopy(current_best)

		# return the best individual
		return self.population.best(fun_evaluation, self.fun_fitness)
		

if __name__ == '__main__':

	from GAComponents import Individual, Population
	from GAOperators import RouletteWheelSelection, RankingSelection, Crossover, Mutation

	# schaffer-N4
	# sol: x=[0,1.25313], min=0.292579
	schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

	ranges = [(-100, 100)] * 2

	I = Individual(ranges)
	P = Population(I, 50)
	SRW = RouletteWheelSelection()
	SR = RankingSelection(0.5)
	C = Crossover([0.5, 0.9], 0.5)
	M = Mutation(0.12)

	g = GA(P, SR, C, M)
	res = g.run(schaffer_n4, 800)	

	x = [0,1.25313] 
	print('{0} : {1}'.format(res.evaluation, res.solution))
	print('error: {:<3f} %'.format((res.evaluation/schaffer_n4(x)-1.0)*100))