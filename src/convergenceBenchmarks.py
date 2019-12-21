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

def ConvergenceVarDamping(matrix,itLimit=100):

	N = matrix.shape[0]

	#Number of iteration
	k = itLimit

	#We generate a simple population vector (1,0,0,0,0,...,0)
	populationVector = np.zeros(N)
	populationVector[0] = 1.0

	pointsNum = 100

	#We are going to compute the evolution of the difference between the exact solution and the Kth iteration
	differencesVector = np.zeros(pointsNum) 

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
matrix = generateRandomStandardizedLinkMatrix(100,True,True)

#iterativeConvergenceConstDamping(matrix,0.85,100)

ConvergenceVarDamping(matrix,100)

#Input in order to keep the plot alive
t = input("Press Enter to finish the programm...")

