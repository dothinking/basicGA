from GAComponents import Individual, Population
from GAOperators import RouletteWheelSelection, RankingSelection, Crossover, Mutation
from GA import GA

from TestFun import *

import time

# objective
f = Himmelblau()

# GA process
s = time.time()
I = Individual(f.ranges)
P = Population(I, 100)
S = RouletteWheelSelection()
C = Crossover([0.6,0.9], 0.55)
M = Mutation(0.1)

g = GA(P, S, C, M)
res = g.run(f.objective,500)

# theoretical result
print('\nGlobal solution input: {0}'.format(f.solution))
print('Global solution output: {0}\n'.format(f.value))

# GA result
print('\nGA solution input: {0}'.format(res.solution))
print('GA solution output: {0}\n'.format(res.evaluation))

print('\nspent time:', time.time()-s)