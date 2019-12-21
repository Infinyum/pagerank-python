import numpy as np
import matplotlib.pyplot as plt

from main import *
from matrix import *
import time

def rankEvolutionDamping(matrix,itLimit=100):

	N = matrix.shape[0]

	pointsNum = 50

	rankPageOne = np.zeros(pointsNum) 

	initDamping = 0.0
	index = 0

	step = (1.0-initDamping) / pointsNum

	damping = initDamping
	while damping < 1.0:

		exactResult  = pageRankMarkov(matrix, damping)
		rankPageOne[index] = exactResult[0]
		#print(rankPageOne[index])

		damping = damping + step
		index = index + 1


	dampingAxis = np.linspace(0,1,pointsNum)

	axes = plt.gca()
	axes.set_xlim([0,1])
	#axes.set_ylim([0, 2*1e-16])

	plt.figure(1)
	plt.plot(dampingAxis,rankPageOne,'r-',label='Rank of the first page')
	plt.legend()
	plt.title("Evolution of the rank of a page according to the damping")       #Adding a title
	plt.xlabel("Damping") 	#X axis title
	plt.ylabel("Rank") 	#Y axis title
	plt.show()


def selfReferencingMatrix(matrix,itLimit=100):

	N = matrix.shape[0]

	#We generate a simple population vector with equal number of surfers on each page
	populationVector = np.zeros(N)
	part = 1.0 / N
	for i in range(0, N):
		populationVector[i] = part

	pointsNum = 50

	# Page one is the one only refering to itself
	rankPageOne = np.zeros(pointsNum) 

	initDamping = 0.0
	index = 0

	step = (1-initDamping) / pointsNum

	damping = initDamping
	while damping < 1.0:

		exactResult  = pageRankMarkov(matrix, damping)
		rankPageOne[index] = exactResult[0]
		#print(rankPageOne[index])

		damping = damping + step
		index = index + 1


	dampingAxis = np.linspace(0,1,pointsNum)

	axes = plt.gca()
	axes.set_xlim([0,1])
	#axes.set_ylim([0, 2*1e-16])

	plt.figure(1)
	plt.plot(dampingAxis,rankPageOne,'g-',label='Rank of the first page')
	plt.legend()
	plt.title("Evolution of the rank of a self-referencing page according to the damping")
	plt.xlabel("Damping")
	plt.ylabel("Rank")
	plt.show()


def selfReferencingMatrixIncremental(matrix,itLimit=100):

	N = matrix.shape[0]

	#We generate a simple population vector with equal number of surfers on each page
	populationVector = np.zeros(N)
	part = 1.0 / N
	for i in range(0, N):
		populationVector[i] = part

	for i in range(0, N):
		matrix[i][0] = part

	pointsNum = 100
	step = 1 / pointsNum
	index = 0
	damping = 0.85
	
	# Page one is the one only refering to itself
	rankPageOne = np.zeros(pointsNum) 

	for _ in range(0, pointsNum):

		exactResult  = pageRankMarkov(matrix, damping)
		rankPageOne[index] = exactResult[0]

		for i in range(0, N):
			matrix[i][0] -= (step / (N-1))
		matrix[0][0] += step
		
		index = index + 1

	selfReferenceAxis = np.linspace(0,1,pointsNum)

	axes = plt.gca()
	axes.set_xlim([0,1])

	plt.figure(1)
	plt.plot(selfReferenceAxis,rankPageOne,'g-',label='Rank of the first page')
	plt.legend()
	plt.title("Evolution of the rank of a self-referencing page according to its self-referencing factor")
	plt.xlabel("Self-referencing factor")
	plt.ylabel("Rank")
	plt.show()


N = 100

matrix = generateRandomStandardizedLinkMatrix(N,True,True)

# The first page only reference itself
#for i in range(0, N):
#	matrix[i][0] = 0
#matrix[0][0] = 1

selfReferencingMatrixIncremental(matrix, 100)
#rankEvolutionDamping(matrix,100)