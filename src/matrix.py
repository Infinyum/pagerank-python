import numpy as np

#Define the ratio of how empty our sparse matrices are
EMPTINESS_RATIO = 0.75
#Define the ratio of how many outside links a website is having in the matrix
LINK_EMPTINESS_RATIO = 0.75

#np.set_printoptions(precision=3)

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
    @parameter size, the size of the vector
    @parameter empty, do we want an "empty" vector (for sparse matrix) => only a very limited amount of not-null component
    @return a probabilistic vector
    """

    res = np.zeros(size)

    if(empty):

        #Generate the indexes to populate
        emptyVecIndexes = np.random.choice(2,size, p=[LINK_EMPTINESS_RATIO,1-LINK_EMPTINESS_RATIO])
        #Generate a random vector of the right size
        randomVec = np.random.random(size)

        #populate those vectors
        res[emptyVecIndexes==1] = randomVec[emptyVecIndexes==1]

    else:
        #Generate an uniform random vector of the right size
        res = np.random.uniform(0,1,size)

    #Normalize the res vector
    total = res.sum()
    if(total == 0):
        return res
    else:
        return res/total


def generateRandomStandardizedLinkMatrix(size, empty, autoRefLink):
    """
    Function that generates a Random Link Matrix for the PageRank problem
    This is equivalent to the S (for Standardized Link) matrix in the algorithm
    @parameter size -> int, the number of WebPage to rank (size of the matrix)
    @parameter emtpy -> bool, Do we use a sparse matrix ?
    @parameter autoRefLink -> bool, Do we authorize link on the same website ?
    @return the NxN sized (randomly generated) matrix
    """

    #We start by generating our matrix
    res = np.zeros((size,size),dtype=float);

    #If we want to work with a sparse matrix
    #We Generate the index vector (witch vector to populate?)
    emptyVecIndexes = np.random.choice(2,size, p=[EMPTINESS_RATIO,1-EMPTINESS_RATIO])


    for i in range(size):

        ## SPARSE MATRIX ##
        if(empty):

            #We generate random vectors for only few columns
            if(emptyVecIndexes[i]==1):
                res[i] = generateProbabilisticVector(size,True)

            #We postprocess the non empty columns to ensure certain properties (diag = 0 | sum = (strict) 1 )
            if(res[i].sum()!=0):
                index = np.random.choice(size,1)

                while(index==i):
                    index = np.random.choice(size,1)


                if(autoRefLink==False):
                    res[i][index]+=res[i][i]
                    res[i][i]=0

                #float precision sum problem => we ensure normalization of columns
                if(isProbabilisticVector(res[i]) == False):
                    diff = 1-res[i].sum()
                    res[i][index]+=diff

            #for vectors with no link => Same chances to go anywhere
            else:
                #fullfill empty vectors with the same prob
                res[i]= np.full(size,1/size)

        ## NORMAL MATRIX ##
        else:
            res[i] = generateProbabilisticVector(size,False)

            #Very unlikely but we do it just to be sure
            if res[i].sum()==0:

                #fullfill empty vectors with the same prob
                res[i]= np.full(size,1/size)


            #We postprocess the non empty columns to ensure certain properties (diag = 0 | sum = (strict) 1 )
            else:
                index = np.random.choice(size,1)

                while(index==i):
                    index = np.random.choice(size,1)

                if(autoRefLink==False):
                    res[i][index]+=res[i][i]
                    res[i][i]=0

                #float precision sum problem => we ensure normalization of columns
                if(isProbabilisticVector(res[i]) == False):
                    diff = 1-res[i].sum()
                    res[i][index]+=diff

    #to remove
    #print(np.transpose(res));
    return np.transpose(res)

#To remove
#generateRandomStandardizedLinkMatrix(7,True,True)
