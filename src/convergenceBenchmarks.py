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

def convergenceVarDamping(matrix,itLimit=100, pointsNum=10):

	#Init part
	N = matrix.shape[0]
	initDamping = 0.1
	step = (1-initDamping) / pointsNum
	damping = initDamping

	#We generate a simple population vector with equal number of surfers on each page
	populationVector = np.zeros(N)
	part = 1.0 / N
	for i in range(0, N):
		populationVector[i] = part



	while damping < 1.0:

		exactResult = pageRankMarkov(matrix, damping)

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

		label = 'damping : ' + str(np.around(damping,2))

		plt.figure(1)
		plt.plot(iterations,differencesVector,label=label)
		plt.legend()
		plt.title("Average Distance to the exact solution depending on the number of iteration")       #Adding a title
		plt.xlabel("Number of iteration") 	#X axis title
		plt.ylabel("Distance") 				#Y axis title

		plt.yscale('log')
		damping = damping + step
		

	plt.show()

def precisionVarDamping(matrix,itLimit=100):

	N = matrix.shape[0]

	#We generate a simple population vector (1,0,0,0,0,...,0)
	populationVector = np.zeros(N)
	populationVector[0] = 1.0

	#We are going to compute the evolution of the difference between the exact solution and the Kth iteration
	differencesVector = np.zeros(pointsNum) 

	pointsNum = itLimit
	initDamping = 0.01
	index = 0

	step = (1-initDamping) / pointsNum

	damping = initDamping
	while damping < 1.0:

		exactResult  = pageRankMarkov(matrix, damping)
		approxResult = pageRankMarkovByStep(matrix, populationVector, damping, k)
		differencesVector[index] = np.average(np.abs(exactResult.T[0]-approxResult));

		damping = damping + step
		index = index + 1


	#PLOT PART

	maxVal = np.amax(differencesVector)

	x = [0,1]
	y = [exactResult,exactResult]

	dampingAxis = np.linspace(0,1,pointsNum)

	axes = plt.gca()
	axes.set_xlim([0,1])
	axes.set_ylim([0, 2*1e-16])

	plt.figure(1)
	plt.plot(dampingAxis,differencesVector,'r-',label='Distance to the exact solution')
	plt.legend()
	plt.title("Average distance to the exact solution depending on the damping factor")       #Adding a title
	plt.xlabel("Damping") 	#X axis title
	plt.ylabel("Distance") 	#Y axis title
	plt.show()


#matrix = createMatrix()
matrix = generateRandomStandardizedLinkMatrix(1000,True,True)

#iterativeConvergenceConstDamping(matrix,0.85,100)

#precisionVarDamping(matrix,100)

convergenceVarDamping(matrix,100,5)

#Input in order to keep the plot alive
t = input("Press Enter to finish the programm...")

