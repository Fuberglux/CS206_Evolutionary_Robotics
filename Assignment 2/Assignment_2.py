'''
Griffin Jones
Assignment 2
Artificial Neural Network
'''
from scipy import *
from numpy import *
import matplotlib as mpl
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
    plt.plot(vector)

def SynapsePlot(neuronPositions,synapses):
    for j in range(0,10):
        for i in range(0,10):
            weight = synapses[i,j]
            if weight > 0:
                polarity = [0,0,0]
            else:
                polarity = [.8,.8,.8]
            weight = int(10*abs(synapses[i,j]))+1
            plt.plot([neuronPositions[0,i],neuronPositions[0,j]],[neuronPositions[1,i],neuronPositions[1,j]],'-',color=polarity,linewidth=weight)

def Update(neuronValues,synapses,i):
    dim = neuronValues.shape
    for j in range(dim[1]):
        temp=synapses[j,0]*neuronValues[i-1,0]
        for k in range(1,10):
            temp=temp+(synapses[j,k]*neuronValues[i-1,k])
        if temp<0:
            temp=0
        if temp>1:
            temp=1
        neuronValues[i,j]=temp
    return(neuronValues)
        
def Main():
    numUpdates = 50
    numNeurons = 10
    neuronValues=MatrixCreate(numUpdates,numNeurons)
    for j in range(numNeurons):
        neuronValues[0,j] = random.random()
    # We have initialized the neural network with random values
    
    synapses = MatrixCreate(numNeurons,numNeurons)
    for k in range(0,10):
        for j in range(0,10):
            synapses[j,k] = random.uniform(-1,1)
    # We have created the synapse matrix
    
    for i in range(1,numUpdates):
        Update(neuronValues,synapses,i)
    # We are populating the neuronValues matrix starting at the second row

'''
Note: This commented block contains everything necessary to
display the ANN in a ring. Otherwise, it is separate.
'''
##    neuronPositions=MatrixCreate(2,numNeurons)
##    angle = 0.0
##    angleUpdate = 2 * pi/numNeurons
##    for i in range(numNeurons):
##        angle = angle + angleUpdate
##        neuronPositions[0,i] = sin(angle)
##        neuronPositions[1,i] = cos(angle)
##    # We have created the positions necessary to display the ANN
##    
##    plt.title('Neural Network')
##    plt.plot(neuronPositions[0,:],neuronPositions[1,:],'ko',markerfacecolor=[1,1,1],markersize=18)
##    SynapsePlot(neuronPositions,synapses)
##    # We have drawn the ANN with weight synapses and neuron values.
    
    plt.imshow(neuronValues,cmap=cm.gray,aspect='auto',interpolation='nearest')
    plt.show()
Main()


