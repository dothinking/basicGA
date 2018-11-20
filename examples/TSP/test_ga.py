import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

# built-in modules from GA Package
from GA.GAPopulation.SequenceIndividual import UniqueLoopIndividual
from GA.GAPopulation.Population import Population
from GA.GAOperators.Selection import RouletteWheelSelection, LinearRankingSelection
from GA.GAOperators.Crossover import SequencePMXCrossover
from GA.GAOperators.Mutation import UniqueSeqMutation
from GA.GAProcess import GA

from TspCities import TSPCities

def test(cities, gen):

	# GA process
	I = UniqueLoopIndividual(cities.dimension)
	P = Population(I, 50)
	L = LinearRankingSelection(500)
	C = SequencePMXCrossover([0.6, 0.9])
	M = UniqueSeqMutation(0.2)
	g = GA(P, L, C, M)

	# solve
	res = g.run(cities.distance, gen)

	return res


if __name__ == '__main__':

	cities = TSPCities('dataset/a280.tsp', 'dataset/a280.opt.tour')

	# build-in GA process
	s1 = time.time()
	res = test(cities, 500)

	# output
	s2 = time.time()
	print('Global solution    : {0}'.format(cities.min_distance))
	print('General GA solution: {0} in {1} seconds'.format(res.evaluation, s2-s1))

	# plot
	cities.plot_cities(plt)
	cities.plot_path(plt, cities.solution)
	cities.plot_path(plt, res.solution)
	plt.show()