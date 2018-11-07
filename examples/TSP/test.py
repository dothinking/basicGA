import os
import sys
import time

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

from GA.GAPopulation.SequenceIndividual import UniqueSeqIndividual
from GA.GAPopulation.Population import Population
from GA.GAOperators.Selection import RouletteWheelSelection
from GA.GAOperators.Crossover import UniqueSeqCrossover
from GA.GAOperators.Mutation import UniqueSeqMutation
from GA.GAProcess import GA

from cities import TSP

import matplotlib.pyplot as plt



s = time.time()

t = TSP('dataset/eil51.tsp', 'dataset/eil51.opt.tour')

# objective function
f = lambda x: t.distances(x)

# GA process
I = UniqueSeqIndividual(t.dimension)
P = Population(I, 100)
R = RouletteWheelSelection()
C = UniqueSeqCrossover([0.6, 0.9])
M = UniqueSeqMutation(0.1)
g = GA(P, R, C, M)

# solve
res = g.run(f, 2000)

# theoretical result
print('Global solution output: {0}'.format(t.min_distance))

# GA result
print('GA solution output: {0}'.format(res.evaluation))

print('spent time: {0}\n'.format(time.time()-s))


# plot
fig = plt.figure()
ax = fig.add_subplot(111)
t.plot_cities(ax)
t.plot_path(ax, t.solution)
t.plot_path(ax, res.solution)
plt.show()