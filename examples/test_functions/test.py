import os
import sys
import time

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

from GA.GAPopulation.DecimalIndividual import DecimalFloatIndividual
from GA.GAPopulation.Population import Population
from GA.GAOperators.Selection import RouletteWheelSelection
from GA.GAOperators.Crossover import DecimalCrossover
from GA.GAOperators.Mutation import DecimalMutation
from GA.GAProcess import GA

from functions import *



def test(FUN):

	s = time.time()

	# objective function
	f = FUN()

	# GA process
	I = DecimalFloatIndividual(f.ranges)
	P = Population(I, 50)
	R = RouletteWheelSelection()
	C = DecimalCrossover([0.6,0.9], 0.55)
	M = DecimalMutation(0.2)
	g = GA(P, R, C, M)

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