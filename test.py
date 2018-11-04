from GAComponents import Individual, Population
from GAOperators import RouletteWheelSelection, LinearRankingSelection, Crossover, Mutation
from GA import GA
from TestFun import *
import time


def test(FUN):

	s = time.time()

	# objective function
	f = FUN()

	# GA process	
	I = Individual(f.ranges)
	P = Population(I, 50)
	R = RouletteWheelSelection()
	L = LinearRankingSelection(99)
	C = Crossover([0.6,0.9], 0.55)
	M = Mutation(0.2)
	g = GA(P, L, C, M)

	# solve
	res = g.run(f.objective, 500)

	# theoretical result
	print('---TEST FUNCTION: {0}---'.format(FUN.__name__))
	print('Global solution input: {0}'.format(f.solution))
	print('Global solution output: {0}'.format(f.value))

	# GA result
	print('GA solution input: {0}'.format(res.solution))
	print('GA solution output: {0}'.format(res.evaluation))

	print('spent time: {0}\n'.format(time.time()-s))

if __name__ == '__main__':

	FUNS = [Bukin, Eggholder, Rosenbrock]
	for fun in FUNS:
		test(fun)