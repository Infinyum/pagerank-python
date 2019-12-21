import numpy as np
import matplotlib.pyplot as plt

from main import *
from matrix import *
import time

def rankEvolutionDamping(matrix,itLimit=100):

	N = matrix.shape[0]

	pointsNum = 50

	#We are going to compute the evolution of the difference between the exact solution and the Kth iteration
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


matrix = generateRandomStandardizedLinkMatrix(100,True,True)

rankEvolutionDamping(matrix,100)