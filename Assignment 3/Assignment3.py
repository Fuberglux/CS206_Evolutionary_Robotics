'''
Griffin Jones
Assignment 3
Using Evolutionary Algorithms with ANNs
'''
from scipy import *
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.spatial.distance as dist

numNeurons=10
numUpdates=10
numGenerations=1000
'''
MATRIX CREATE:
'''
def MatrixCreate(row, col):
    v=zeros((row,col)) 
    return(v)

'''
MATRIX RANDOMIZE:
This differs from the original Matrix Randomize
in that it randomizes from (-1,1) instead of (0,1).
'''
def MatrixRandomize(v): 
    for element in range(v.size):
        v.flat[element:]=random.uniform(-1,1)
    return(v)

'''
VECTOR CREATE:
same functionality as Matrix Create but creates
a row (horizonatal) vector of size j.
'''
def VectorCreate(j):
    v=zeros((j),dtype='f')
    return(v)

'''
UPDATE:
'''
def Update(neuronValues,synapses):
    dim = neuronValues.shape
    for newRow in range(1,10):
        for newNeur in range(10):
            for oldNeur in range(10):
                neuronValues[newRow,newNeur] = neuronValues[newRow,newNeur]+(neuronValues[newRow-1,oldNeur]*synapses[newNeur,oldNeur])
            if neuronValues[newRow,newNeur]<0:
                neuronValues[newRow,newNeur]=0
            if neuronValues[newRow,newNeur]>1:
                neuronValues[newRow,newNeur]=1
    return(neuronValues)

'''
MEAN DISTANCE:
'''
def MeanDistance(v1,v2):
    acc=VectorCreate(numNeurons)
    for j in range(numNeurons):
        d=dist.euclidean(v1[j],v2[j])
        acc[j]=d
    d=sum(acc)/numNeurons
    return(d)

'''
FITNESS:
'''
def Fitness(synapses,currentGeneration): 
    neuronValues=MatrixCreate(numUpdates,numNeurons) #neuronValue matrix
    for j in range(numNeurons): #initialize NV Matrix
        neuronValues[0,j]=0.5
    Update(neuronValues,synapses)
    if currentGeneration == 999:
        plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
        plt.show()
##    plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
##    plt.show()    
##    raw_input('Print ENTER to continue.');
    actualNeuronValues=VectorCreate(numNeurons)#Create vector for storing last row
    actualNeuronValues=neuronValues[9,:] #Save last row of neuronValues
##    print actualNeuronValues
##    raw_input('Print ENTER to continue.');
    desiredNeuronValues=VectorCreate(numNeurons)#Create Ideal Fitness
    for j in range(0,10,2):
        desiredNeuronValues[j]=1
##    print desiredNeuronValues
##    raw_input('Print ENTER to continue.');
##    plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
##    plt.show()
##    print abs(actualNeuronValues-desiredNeuronValues)
##    raw_input('Print ENTER to continue.');
##    print sum(abs(actualNeuronValues-desiredNeuronValues))
##    raw_input('Print ENTER to continue.');
##    print sum(abs(actualNeuronValues-desiredNeuronValues))/10
##    raw_input('Print ENTER to continue.');
    return(1-sum(abs(actualNeuronValues-desiredNeuronValues))/10)

'''
FITNESS2:
'''
def Fitness2(synapses,currentGeneration): 
    neuronValues=MatrixCreate(numUpdates,numNeurons) #neuronValue matrix
    for j in range(numNeurons): #initialize NV Matrix
        neuronValues[0,j]=0.5
    Update(neuronValues,synapses)
    if currentGeneration == 999:
        plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
        plt.show()
##    plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
##    plt.show()    
##    raw_input('Print ENTER to continue.');
    actualNeuronValues=VectorCreate(numNeurons)#Create vector for storing last row
    actualNeuronValues=neuronValues[9,:] #Save last row of neuronValues
##    print actualNeuronValues
##    raw_input('Print ENTER to continue.');
    desiredNeuronValues=VectorCreate(numNeurons)#Create Ideal Fitness
    for j in range(0,10,2):
        desiredNeuronValues[j]=1
##    print desiredNeuronValues
##    raw_input('Print ENTER to continue.');
##    plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
##    plt.show()
##    print abs(actualNeuronValues-desiredNeuronValues)
##    raw_input('Print ENTER to continue.');
##    print sum(abs(actualNeuronValues-desiredNeuronValues))
##    raw_input('Print ENTER to continue.');
##    print sum(abs(actualNeuronValues-desiredNeuronValues))/10
##    raw_input('Print ENTER to continue.');
    diff=0.0
    for i in range(0,9):
        for j in range(0,9):
            diff=diff + abs(neuronValues[i,j]-neuronValues[i,j+1])
            diff=diff + abs(neuronValues[i+1,j]-neuronValues[i,j])
    diff=diff/(2*9*9)
    return(diff)
'''
MATRIX PERTURB:
'''
def MatrixPerturb(parent,prob):
    random.seed()
    dim=parent.shape
    child=MatrixCreate(numNeurons,numNeurons) # For a 2-D Matrix
    for j in range(numNeurons):
        for i in range(numNeurons):
            child[i,j]=parent[i,j]
    rval=random.uniform(-1,1)
    for j in range(numNeurons):
        for i in range(numNeurons):
            if (rval < prob): #is the random value lower than the set limit?
                child[i,j] = random.uniform(-1,1) #if so, reassign the current value [0,1)
            rval = random.rand()
    return(child)

'''
Main Function:
'''
def main():
    fits=VectorCreate(numGenerations)
    parent=MatrixCreate(numNeurons,numNeurons) #synapse matrix
    parent=MatrixRandomize(parent) #randomized synapse matrix
    parentFitness=Fitness(parent,0)
    for currentGeneration in range(numGenerations):
        child = MatrixPerturb(parent, 0.05)
        childFitness = Fitness(child,currentGeneration)    
        if(childFitness>parentFitness):
            parent = child
            parentFitness = childFitness
        fits[currentGeneration] = parentFitness
    plt.plot(fits, '-')
    plt.title('Fitness Curve')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

main()
