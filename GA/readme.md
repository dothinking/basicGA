# Practise Genetic Algorithm in Python

## GA Process

A general process is implemented to solve different problems, e.g. `mathematical function`, `TSP`, which would be modeled with built-in or user-defined Individals and Operators. Features of this process:

- single-objective minimization
- elitist preservation to improve Simple GA
- adaptive crossover probability

## Population

A common Population is implemented based on built-in or used-defined Individuals.

### Built-in Individuals

- `DecimalFloatIndividual` for problems with float solutions, e.g. multivariate function
- `DecimalIntegerIndividual` for problems with integer solutions, e.g. multivariate function

- `UniqueSeqIndividual` for problems with a sequence solution
- `UniqueLoopIndividual` for problems with a close loop sequence solution, e.g. travelling salesman problem
- `ZeroOneSeqIndividual` for problems with a 0-1 sequence solution


### User-defined Individuals

It should be derived from Base class `Individual` and override `init_solution(self,ranges)` to define how to create a random initial solution.

```python
class UserDefinedIndividual(Individual):
    '''user defined individual'''
    
    def init_solution(self, ranges):
        '''initialize random solution in `ranges`
        three required properties:
        - self._ranges: ranges for the solution
        - self._dimension: count of variables
        - self._solution: solution for the problem
        '''
        pass
```

## Operators

Three kinds of operators, `Selection`, `Crossover`, `Mutation`, are considered. Except `Selection` operator, **`Crossover` and `Mutation` should be compatible with the applied individual**, which is defined by the property `self._individual_class` and checked in `GAProcess`.

### Built-in Operators

- `RouletteWheelSelection`: select individuals by Roulette Wheel with a probability of their fitness
- `LinearRankingSelection`: select individuals by Roulette Wheel with a probability of their ranking postions
- `TournamentSelection`: select individuals by tournament

- `DecimalCrossover`: linear interpolation for decimal encoded individuals
- `SequencePMXCrossover`: Partially Mapped Crossover for unique sequence individuals
- `SequenceOXCrossover`: Order Crossover for unique sequence individuals

- `DecimalMutation`: add random deviations for decimal encoded individuals
- `UniqueSeqMutation`: exchange genes for unique sequence individuals


### user-defined Operators

Derived from `Selection` and override `select(self, population)` to define the individuals to be selected.

```python
class UserDefinedSelection(Selection):
    def select(self, population):
        '''
        - population: where the individuals from
        - return: the selected individuals
        '''
        raise NotImplementedError
```

Derived from `Crossover` and override `cross_individuals(individual_a, individual_b, pos, alpha)` to define how to create new individuals from the selected two individuals. Besides, the valid Individual class name should be defined in property `self._individual_class`.

```python
class UserDefinedCrossover(Crossover):
    def __init__(self):
        super().__init__()
        self._individual_class = [...]

    def cross_individuals(individual_a, individual_b, pos, alpha):
        '''
        generate two individuals based on parent individuals:
            - individual_a, individual_b: the selected individuals
            - pos  : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: two generated individuals
        '''
        raise NotImplementedError
```


Derived from `Mutation` and override `mutate_individual(individual, positions, alpha)` to define how to create new individuals from the selected two individuals. Besides, the valid Individual class name should be defined in property `self._individual_class`.

```python
class UserDefinedMutation(Mutation):
    def __init__(self):
        super().__init__()
        self._individual_class = [...]

    def mutate_individual(individual, positions, alpha):
        '''
        get mutated solution based on the selected individual:
            - individual: the selected individual
            - positions : 0-1 vector to specify positions for crossing
            - alpha: additional param
            - return: the mutated solution
        '''
        raise NotImplementedError
```
