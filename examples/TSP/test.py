import os
import sys
import time

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

from GA.GAPopulation.SequenceIndividual import UniqueLoopIndividual
from GA.GAPopulation.Population import Population
from GA.GAOperators.Selection import RouletteWheelSelection, LinearRankingSelection
from GA.GAOperators.Crossover import UniqueSeqCrossover
from GA.GAOperators.Mutation import UniqueSeqMutation
from GA.GAProcess import GA

from cities import TSP
import matplotlib.pyplot as plt
import numpy as np


s = time.time()

t = TSP('dataset/eil51.tsp', 'dataset/eil51.opt.tour')
# t = TSP('dataset/a10.tsp')
# t = TSP('dataset/a280.tsp')

# objective function
f = lambda x: t.distances(x)

# GA process
I = UniqueLoopIndividual(t.dimension)
P = Population(I, t.dimension*10)
R = RouletteWheelSelection()
L = LinearRankingSelection(250)
C = UniqueSeqCrossover([0.5, 0.9])
M = UniqueSeqMutation(0.15)
g = GA(P, L, C, M)

# solve
res = g.run(f, 10000)
print('Global solution output: {0}'.format(t.min_distance)) # theoretical result
print('GA solution output: {0}'.format(res.evaluation)) # GA result
print('GA solution output: {0}'.format(res.solution)) # GA result
print('spent time: {0}\n'.format(time.time()-s))


# plot
t.plot_cities(plt)
t.plot_path(plt, t.solution)
t.plot_path(plt, res.solution)
plt.show()