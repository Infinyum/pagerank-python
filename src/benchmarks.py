import numpy as np
import matplotlib.pyplot as plt

from main import *
from matrix import *
import time

#Convention : 	function_sm => for sparse matrices
#				function_m => for the other matrices

def iterativeConvergenceConstDamping(matrix,damping=0.85,itLimit=100,):

	N = matrix.shape[0]
	exactResult = pageRankMarkov(matrix, damping)

	#populationVector = generateProbabilisticVector(N, False)

	#We generate a simple population vector (1,0,0,0,0,...,0)
	populationVector = np.zeros(N)
	populationVector[0] = 1.0

	#We are going to compute the evolution of the difference between the exact solution and the Kth iteration
	differencesVector = np.zeros(100) 

	k = itLimit

	#Compute the kth iterations 
	while(k>0):

		populationAfterKIteration = pageRankMarkovByStep(matrix, populationVector, damping, k)
		differencesVector[k-1] = np.average(np.abs(exactResult.T[0] - populationAfterKIteration))
		k = k - 1

	#PLOT PART

	maxVal = np.amax(differencesVector)

	x = [0,itLimit]
	y = [exactResult,exactResult]

	iterations = np.linspace(0,itLimit,100)

	axes = plt.gca()
	axes.set_xlim([0,itLimit+2])
	#axes.set_ylim([0, maxVal])

	plt.figure(1)
	plt.plot(iterations,differencesVector,'b-',label='Distance to the exact solution')
	plt.legend()
	plt.title("Average Distance to the exact solution depending on the number of iteration")       #Adding a title
	plt.xlabel("Number of iteration") 	#X axis title
	plt.ylabel("Distance") 				#Y axis title

	plt.yscale('log')

	plt.show()





#matrix = createMatrix()
#matrix = generateRandomStandardizedLinkMatrix(100,False,True)

#iterativeConvergenceConstDamping(matrix,0.85,100)

#Input in order to keep the plot alive
#t = input("Press Enter to finish the programm...")

if __name__ == "__main__":
	N = 10000
	matrix = generateRandomStandardizedLinkMatrix(N, False, True)

	setup1 = """from main import pageRankMarkovByStep
from __main__ import N, matrix
import numpy as np

damping = 0.85
k = 10

initialDistribution = np.zeros(N)
initialDistribution[0] = 1.0
"""

	setup2 = """from main import pageRankMarkovByStep
from __main__ import N, matrix
import numpy as np

damping = 0.85
k = 100

initialDistribution = np.zeros(N)
initialDistribution[0] = 1.0
"""

	setup3 = """from main import pageRankMarkovByStep
from __main__ import N, matrix
import numpy as np

damping = 0.85
k = 10

initialDistribution = np.zeros(N)
part = 1.0 / N
for i in range(0, N):
    initialDistribution[i] = part
"""

	setup4 = """from main import pageRankMarkovByStep
from __main__ import N, matrix
import numpy as np

damping = 0.85
k = 100

initialDistribution = np.zeros(N)
part = 1.0 / N
for i in range(0, N):
    initialDistribution[i] = part
"""

	import timeit

	print("k = 10, all in the first page")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup1, number=1))

	print("\nk = 100, all in the first page")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup2, number=1))

	print("\nk = 10, all spread equally on all pages")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup3, number=1))

	print("\nk = 100, all spread equally on all pages")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup4, number=1))