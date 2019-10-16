import numpy as np

def createMatrix():
    line1 = np.array([0, 0.5, 0.2, 0, 0])
    line2 = np.array([0, 0, 0.2, 0, 0])
    line3 = np.array([1, 0.5, 0.2, 0, 0])
    line4 = np.array([0, 0, 0.2, 0, 1])
    line5 = np.array([0, 0, 0.2, 1, 0])
    
    matrix = np.array([line1, line2, line3, line4, line5])
    
    return matrix


def getOutlink(matrix):
    N = matrix.shape[0]
    outlink = np.zeros(shape=(N, 1))

    for i in range(0, N):
        outlink[i] = np.count_nonzero(matrix.T[i])
    
    return outlink
    


def pageRank(matrix, k=0, damping=0.85):
    # The number of rows in the matrix i.e. the number of pages
    N = matrix.shape[0]

    # Initialize the page rank of each page with equi-probability
    equiProbability = 1.0 / N 
    rank = np.full((N, 1), equiProbability)

    # Get the number of outlinks of each page from the matrix
    outlink = getOutlink(matrix)

    # PageRank algorithm running for k iterations
    for _ in range(0, k):
        newRank = np.zeros((N, 1))
        # For each page
        for i in range(0, N):
            newRank[i] = ((1 - damping) / N) + (damping * sum(rank[j] / outlink[j] for j in range(0, N)))
        rank, newRank = newRank, rank
    
    print(rank)
    print(np.sum(rank))


if __name__ == "__main__":
    k = 1
    damping = 0.85
    matrix = createMatrix()
    pageRank(matrix, k, damping)