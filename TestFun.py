# test functions for single-objective optimization cases
import numpy as np	

class FUN:
	def __init__(self, n=None):
		self.objective = None
		self.ranges = None
		self.solution = None

	@property
	def value(self):
		return self.objective(self.solution[0])


class Beale(FUN):
	def __init__(self):
		self.objective = lambda x: (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2
		self.ranges = [(-4.5,-4.5), (4.5,4.5)]
		self.solution = [(3,0.5)]

if __name__ == '__main__':
	from GA import GA
	from GAComponents import Individual, Population
	from GAOperators import RouletteWheelSelection, RankingSelection, Crossover, Mutation

	f = Beale()
	I = Individual(f.ranges)
	P = Population(I, 50)
	S1 = RouletteWheelSelection()
	S2 = RankingSelection(0.33)
	C = Crossover([0.5,0.9], 0.5)
	M = Mutation(0.08)

	g = GA(P, S2, C, M, lambda x:np.exp(-x))
	res = g.run(f.objective, 50)

	# theoretical res
	print('\nGlobal solution input: {0}'.format(f.solution))
	print('Global solution output: {0}\n'.format(f.value))

	print('\nGA solution input: {0}'.format(res.solution))
	print('GA solution output: {0}\n'.format(res.evaluation))
