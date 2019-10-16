import numpy as np

#Define the ratio of how empty our sparse matrices are
EMPTINESS_RATIO = 0.75
#Define the ratio of how many link our website matrices have
LINK_EMPTINESS_RATIO = 0.75


def isProbabilisticVector(vec):
    """
    Function that states if a vector is probabilistic or not (sum of its components = 1)
    @parameter vec, the vector to check
    @return a boolean that states if the vector is probabilistic
    """
    return sum(vec)==1

def generateProbabilisticVector(size, empty):
    """
    Function that generates a probabilistic vector (sum of its components = 1)
    TODO
    """

    res = np.zeros(size)

    #Generates the vector index to populate
    emptyVecIndexes = np.random.choice(2,size, p=[LINK_EMPTINESS_RATIO,1-LINK_EMPTINESS_RATIO])
    randomVec = np.random.random(size)

    #populate those vectors
    res[emptyVecIndexes==1] = randomVec[emptyVecIndexes==1]
    total = res[emptyVecIndexes==1].sum()
    if(total == 0):
        return res
    else:
        return res/total


def generateRandomLinkMatrix(size, empty):
    """
    Function that generates a Random Link Matrix for the PageRank problem
    This is equivalent to the L (for Link) matrix in the algorithm
    @parameter size -> int, the number of WebPage to rank (size of the matrix)
    @parameter emtpy -> bool, Do we use a sparse matrix ?
    @return the NxN sized random generated matrix
    """

    #We start by generating our matrix
    res = np.zeros((size,size),dtype=float);

    #If we want to work with a sparse matrix
    if(empty):

        #Generates the vector index to populate
        emptyVecIndexes = np.random.choice(2,size, p=[EMPTINESS_RATIO,1-EMPTINESS_RATIO])
        #populate those vectors
        res[emptyVecIndexes==1] = generateProbabilisticVector(size,empty)

    else:
        print("TODOD")

    #to remove
    print(np.transpose(res));
    return np.transpose(res)

#To remove
generateRandomLinkMatrix(10,True)
