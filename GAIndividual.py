#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np


class Individual():
	'''individual of population'''
	def __init__(self, dim, lbound, ubound):
		'''dimension of individual'''
		self._dim = dim
		self._lbound = lbound
		self._ubound = ubound
		self._chrom = np.empty((dim,))
		
		self.evaluation = None
		self.fitness = None

	def initialize(self):
		'''initialize random values in [lbound, ubound]'''
		seeds = np.random.random(self._dim)		
		self._chrom = self._lbound + (self._ubound-self._lbound)*seeds

	@property
	def dimension(self):
		return self._dim

	@property
	def bound(self):
		return self._lbound, self._ubound

	@property
	def chrom(self):
		return self._chrom	

	@chrom.setter
	def chrom(self, chrom):
		assert self.dimension == chrom.shape[0]
		assert (chrom>=self._lbound).all() and (chrom<=self._ubound).all()
		self._chrom = chrom

	def mutate(self, positions, alpha):
		'''
		positions: mutating gene positions, list
		alpha: mutatation magnitude
		'''
		for pos in positions:
			if np.random.rand() < 0.5:
				self._chrom[pos] -= (self._chrom[pos]-self._lbound[pos])*alpha
			else:
				self._chrom[pos] += (self._ubound[pos]-self._chrom[pos])*alpha
				
		self.evaluation = None # reset evaluation
		

if __name__ == '__main__':

	dimension = 6
	lbound = np.array([-10]*dimension)
	ubound = np.array([10]*dimension)

	I = Individual(dimension, lbound, ubound)
	I.initialize()

	print(I.chrom)

	I.mutate([1,3,5], 0.5)

	print(I.chrom)