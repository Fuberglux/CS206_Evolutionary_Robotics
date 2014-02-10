from numpy import *
from scipy import *
import matplotlib.pyplot as plt
import random

def MatrixCreate(Rows,Columns):
    arr = zeros([Rows,Columns]) #Zeros() creates an array within an array so it must be copied again using list() to get a 1 dimensional array
    arr2 = (arr[0])
    return arr2
    

def MatrixRandomize(ArrayName):
    random.seed()
    arraySize = size(ArrayName)
    for element in range(arraySize):
        ArrayName[element] = random.random()
    return ArrayName  

def Fitness(ArrayName):
    x = 0
    j = 0
    while x<(size(ArrayName)):
        yu = ArrayName[x]
        j = j + yu
        x += 1

    i = j/(size(ArrayName))

    return i

def MatrixPerturb(Parent,Probability):
    random.seed()
    p = MatrixCreate(1,size(Parent))
    for each in range(size(Parent)):
        p[each] = Parent[each]
        
    xx = random.random()
    yy = size(p)
    xy = 0

    while xy<yy:
        if(xx<Probability):
            p[xy] = random.random()
            
        xx = random.random()
        xy += 1
        
    return p

def HillClimber(Generations):
    parent = MatrixCreate(1,Generations)
    parent = MatrixRandomize(parent)
    parentFitness = Fitness(parent)
    graph = MatrixCreate(1,Generations)
    for currentGeneration in range(Generations):
        print currentGeneration, parentFitness
        child = MatrixPerturb(parent,0.05)
        childFitness = Fitness(child)
        if(childFitness>parentFitness):
            parent = child
            parentFitness = childFitness
        graph[currentGeneration] = parentFitness

    return graph

def PlotVectorAsLine(ArrayName):
    ii = plt.plot(ArrayName)
    #plt.show()
    return ii

#####MAIN
def PlotHC(Generations,Lineages):
    for yy in range(0,Lineages):
        g = HillClimber(Generations)
        ii = PlotVectorAsLine(g)
    
    plt.show(ii)
#####MAIN
    
def Genes(Columns,Rows):
    matrix = []
    for i in range(Rows):
        matrix.append(MatrixCreate(1,Columns))
    for x in range(Rows):
        hc_array=HillClimber(Colums)
        
   


        






    
    
    
