from matrix import *

if __name__ == "__main__":
	N = 10000
	matrix = generateRandomStandardizedLinkMatrix(N, True, True)

	setup1 = """from main import pageRankMarkovByStep, pageRankMarkov
from __main__ import N, matrix
import numpy as np

damping = 0.85
k = 30

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

	#print("k = 10, all in the first page")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup1, number=1))
	#print(timeit.timeit("pageRankMarkov(matrix, k)", setup=setup1, number=1))

	"""print("\nk = 100, all in the first page")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup2, number=1))

	print("\nk = 10, all spread equally on all pages")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup3, number=1))

	print("\nk = 100, all spread equally on all pages")
	print(timeit.timeit("pageRankMarkovByStep(matrix, initialDistribution, damping, k)", setup=setup4, number=1))"""