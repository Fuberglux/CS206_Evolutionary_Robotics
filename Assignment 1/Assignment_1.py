'''
Griffin Jones
Assignment 1
Hill Climber
'''

from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def MatrixCreate(row, col):
    v = zeros((row,col)) 
    return(v)

def MatrixRandomize(v):
    for element in range(v.size):
        v.flat[element:] = random.rand()
    return(v)

def Fitness(v):
    size = len(v.flat)
    total = sum(v.flat)
    mean = total/size
    return(mean)

def MatrixPerturb(parent,prob):
    random.seed()
    dim = parent.shape
    child = MatrixCreate(dim[0],dim[1]) # For a 2-D Matrix
    for each in range(len(parent.flat)):
        child.flat[each] = parent.flat[each]

    rval = random.rand()
    z = 0

    while z < child.size:
        if (rval < prob): #is the random value lower than the set limit?
            child[:,z] = random.rand() #if so, reassign the current value [0,1)

        rval = random.rand()
        z += 1
    return(child)

def PlotVectorAsLine(vector):
    mpl.pyplot.plot(vector)
    
def Main():
'''
This first block is a test statement.
Uncomment it to see if the hillclimber is working.
It should print a long list of increasing fitness scores that approach 1.00
'''
##    fits = MatrixCreate(1,5000)
##    parent = MatrixCreate(1,50)
##    parent = MatrixRandomize(parent)
##    parentFitness = Fitness(parent)
##    for currentGeneration in range(0,5000):
##        print currentGeneration, parentFitness
##        child = MatrixPerturb(parent, 0.05)
##        childFitness = Fitness(child)
##        if(childFitness > parentFitness):
##            parent = child
##            parentFitness = childFitness

'''
This second block will run one hill climber
and plot the fitness levels over 5000 generations.
'''
##    fits = MatrixCreate(1,5000)
##    parent = MatrixCreate(1,50)
##    parent = MatrixRandomize(parent)
##    parentFitness = Fitness(parent)
##    for currentGeneration in range(0,5000):
##        child = MatrixPerturb(parent, 0.05)
##        childFitness = Fitness(child)
##        if(childFitness > parentFitness):
##            parent = child
##            parentFitness = childFitness
##        fits[:,currentGeneration] = parentFitness
##
##    plt.plot(fits[0,:], '-')
##    plt.title('Fitness Curves for a Single Run')
##    plt.xlabel('Generation')
##    plt.ylabel('Fitness')
##    plt.show()

'''
This third block will run five hill climbers
and plot the fitness levels over 5000 generations.
'''
##    for trialNumber in range(0,5):
##        fits = MatrixCreate(1,5000)
##        parent = MatrixCreate(1,50)
##        parent = MatrixRandomize(parent)
##        parentFitness = Fitness(parent)
##        for currentGeneration in range(0,5000):
##            child = MatrixPerturb(parent, 0.05)
##            childFitness = Fitness(child)
##            if(childFitness > parentFitness):
##                parent = child
##                parentFitness = childFitness
##            fits[:,currentGeneration] = parentFitness
##
##        plt.plot(fits[0,:], '-')
##    plt.title('Fitness Curves for Five Run')
##    plt.xlabel('Generation')
##    plt.ylabel('Fitness')
##    plt.show()
    
'''
This fourth block will create a gene matrix showing the improvement
of each individual genotype over the course of 5000 generations.

dim is used to get the size of a dimension.
this indexes at 1.
dim[0] -> rows (50)
dim[1] -> columns (5000)
'''
    Genes = MatrixCreate(50,5000)
    dim = Genes.shape
    parent = MatrixCreate(1,50)
    parent = MatrixRandomize(parent)
    parentFitness = Fitness(parent)
    # We have created a parent
    for j in range(0,5000):
        child = MatrixPerturb(parent,0.05)
        childFitness = Fitness(child)
        # We have created a child from the parent
        if ( childFitness > parentFitness ):
            parent = child
            parentFitness = childFitness
        # We have determined the new parent
        for i in range(50): 
            Genes[:,j] = parent;
    plt.imshow(Genes,cmap=cm.gray,aspect='auto',interpolation='nearest')
    plt.title('Generational Improvement')
    plt.xlabel('Generation')
    plt.ylabel('Genes')
    plt.show()

Main()


