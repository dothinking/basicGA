# Practise Genetic Algorithm in Python

- single-objective minimization
- elitist preservation to improve Simple GA
- adaptive crossover propobility
- based on Numpy

## import modules

GA operators could be extended to implement user defined algorithm.

```python
from GAComponents import Individual, Population
from GAOperators import RouletteWheelSelection, Crossover, Mutation
```

## initialize GA

```python
# individual
dimension = 2
ranges = [(-10, 10)] * dimension
I = Individual(ranges)

# population
P = Population(I, 50)

# operators
S = RouletteWheelSelection()
C = Crossover([0.5, 0.9], 0.5) # adaptive crossover rate
# C = Crossover(0.9, 0.5) # constant crossover rate
M = Mutation(0.12)

# GA
g = GA(P, SRW, C, M)
```

## solve

```python
# schaffer-N4
# sol: x=[0,1.25313], min=0.292579
schaffer_n4 = lambda x: 0.5 + (np.cos(np.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

res = g.run(schaffer_n4, gen=800)   

x = [0,1.25313]
print('{0} : {1}'.format(res.evaluation, res.solution)) # 0.29257882535592317 : [1.25339239e+00 6.28576519e-05]
print('error: {:<3f} %'.format((res.evaluation/schaffer_n4(x)-1.0)*100)) # error: 0.000066 %
```
