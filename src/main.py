import numpy as np
from matrix import *
from scipy.sparse.linalg import eigs

def createMatrix():
    line1 = np.array([0, 0.5, 0.2, 0, 0])
    line2 = np.array([0, 0, 0.2, 0, 0])
    line3 = np.array([1, 0.5, 0.2, 0, 0])
    line4 = np.array([0, 0, 0.2, 0, 1])
    line5 = np.array([0, 0, 0.2, 1, 0])
    
    matrix = np.array([line1, line2, line3, line4, line5])
    
    return matrix


def getOutlink(matrix):
    """
    Take a link matrix and return a vector containing the number of outlinks of each page.
    """
    N = matrix.shape[0]
    outlink = np.zeros(shape=(N, 1))

    for i in range(0, N):
        outlink[i] = np.count_nonzero(matrix.T[i])
    
    return outlink



def getInlink(matrix):
    """
    Take a link matrix and return a vector containing the number of inlinks of each page.
    """
    inlink = {}
    
    for i in range(0, matrix.shape[0]):
        inlink[i] = np.flatnonzero(matrix[i])

    return inlink


def pageRank(matrix, k=1, damping=0.85):
    """
    Page rank algorithm: take a link matrix and return a vector containing the rank of each page.

    Arguments:
    matrix -- the link matrix
    k -- the number of iterations for the page rank algorithm (default 1)
    damping -- the probability a user continues to click on further links (default 0.85)
    """
    # The number of rows in the matrix i.e. the number of pages
    N = matrix.shape[0]

    # Initialize the page rank of each page with equi-probability
    equiProbability = 1.0 / N 
    rank = np.full((N, 1), equiProbability)

    # Get the number of outlinks of each page from the matrix
    outlink = getOutlink(matrix)
    # Get the indexes of the pages pointing to each page
    inlink = getInlink(matrix)

    # PageRank algorithm running for k iterations
    while k > 0:
        newRank = np.zeros((N, 1))
        
        # For each page
        for i in range(0, N):
            newRank[i] = ((1 - damping) / N) + (damping * sum(rank[j] / outlink[j] for j in inlink[i]))
        
        rank, newRank = newRank, rank
        k = k - 1

    return rank
    

def pageRankMarkov(matrix, damping=0.85):
    """
    Page rank algorithm: take a link matrix and return a vector containing the rank of each page.
    The rank corresponds to the 

    Arguments:
    matrix -- the link matrix
    damping -- the probability a user continues to click on further links (default 0.85)
    """
    # The number of rows in the matrix i.e. the number of pages
    N = matrix.shape[0]
    
    increment = (1.0 - damping) / N
    
    # Calculate the Markov matrix
    M = (matrix * damping) + increment

    # Get the eigenvector for eigenvalue 1
    _,v = eigs(M, k=1, sigma=1)

    # Normalize vector
    v = np.real(v)
    v = v / np.sum(v)

    return v


def pageRankMarkovByStep(matrix, vec, damping=0.85, k=1):
    """
    Page rank algorithm: take a link matrix and return a vector containing the distribution of the surfers population after k iterations

    Arguments:
    matrix -- the link matrix
    vec -- the initial distribution of surfers in the Markov chain
    damping -- the probability a user continues to click on further links (default 0.85)
    """
    # The number of rows in the matrix i.e. the number of pages
    N = matrix.shape[0]
    
    increment = (1.0 - damping) / N
    
    # Calculate the Markov matrix
    M = (matrix * damping) + increment

    while k > 0:
        vec = M.dot(vec)
        k = k - 1

    return vec


if __name__ == "__main__":
    N = 10000
    k = 10
    damping = 0.85
    matrix = generateRandomStandardizedLinkMatrix(N, False, True)

    initialDistribution = np.zeros(N)
    part = 1.0 / N
    for i in range(0, N):
        initialDistribution[i] = part

    endDistribution = pageRankMarkov(matrix, damping)
    distributionAfterK = pageRankMarkovByStep(matrix, initialDistribution, damping, k)
    
    print("After " + str(k) + " iterations:\n")
    print(distributionAfterK)
    print("\n\nAfter an infinite number of iterations:\n")
    print(endDistribution.T[0])   