from GAComponents import Individual, Population
from GAOperators import RouletteWheelSelection, RankingSelection, Crossover, Mutation
from GA import GA

from TestFun import *

import time

# objective
f = Bukin()

# GA process
s = time.time()
I = Individual(f.ranges)
P = Population(I, 50)
S1 = RouletteWheelSelection()
S2 = RankingSelection(0.33)
C = Crossover([0.6,0.9], 0.55)
M = Mutation(0.1)

g = GA(P, S2, C, M)
res = g.run(f.objective,500)

# theoretical result
print('\nGlobal solution input: {0}'.format(f.solution))
print('Global solution output: {0}\n'.format(f.value))

# GA result
print('\nGA solution input: {0}'.format(res.solution))
print('GA solution output: {0}\n'.format(res.evaluation))

print('\nspent time:', time.time()-s)