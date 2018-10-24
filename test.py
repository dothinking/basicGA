from GA import GA
import math
import numpy as np

# ------------------
# testing functions
# ------------------
# [-2,1]
f1 = lambda x: x[0]*math.sin(10*np.pi*x[0])+2

# sol: x=[0,0], min=0
f2 = lambda x : 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*np.pi*x[0])+math.cos(2*np.pi*x[1]))

# schaffer
# sol: x=[0,0], min=0
schaffer = lambda x: 0.5 + (math.sin((x[0]**2+x[1]**2)**0.5)**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# schaffer-N4
# sol: x=[0,1.25313], min=0.292579
schaffer_n4 = lambda x: 0.5 + (math.cos(math.sin(abs(x[0]**2-x[1]**2)))**2-0.5) / (1.0+0.001*(x[0]**2+x[1]**2))**2

# shubert
# sol: x=[], min=
def shubert(x):
	a = [i*math.cos((i+1)*x[0]+i) for i in range(1,6)]	 
	b = [i*math.cos((i+1)*x[1]+i) for i in range(1,6)]
	return sum(a) * sum(b)


# Rosenbrock: -5.12<=x<=5.12
# sol: x=[1,1], min=0
rosenbrock = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2

# ------------------
# test GA
# ------------------
def test(obj, sol):
	# GA res
	kw = {
		'lbound': [-100, -100],
		'ubound': [100, 100],
		'size'	: 195,
		'max_generation': 30,
		'fitness': lambda x: np.exp(-x),
		# 'selection_mode': 'elite',
		'crossover_rate': 0.85,		
		'crossover_alpha': 0.75,
		'mutation_rate'	: 0.02
	}
	g = GA(obj, 2, **kw)
	_,res = g.solve()

	# theoretical res
	res0 = obj(sol)
	print('\nGlobal solution input: {0}'.format(sol))
	print('Global solution output: {0}\n'.format(res0))

	print('Relative error: {0} %'.format((res-res0)/res0*100))


if __name__ == '__main__':

	# test(f2, [0,0])
	# test(schaffer, [0,0])
	test(schaffer_n4, [0,1.25313])
	# test(schaffer, [0,0])
	# test(rosenbrock, [1,1])
