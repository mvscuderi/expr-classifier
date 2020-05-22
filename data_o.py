from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np 
import torch
import math
import heapq
import sys
import time

# import data from .mat file and format using PyTorch
data_import = loadmat('data.mat')
data = torch.from_numpy(data_import['face'])
data = data.permute(2, 0, 1)
data = data.numpy()
data = data.reshape(600,24*21)
data = data + 2.0 #make all positive for covariance matrix

# constants
training_size = 507
total_objects = 600
num_classes = 3
num_features = 504
kneighbors = 5

#initialize class labels
labels = np.zeros(total_objects, dtype=int) #class labels
num=0
for i in range(0, total_objects):
    if (num > (num_classes-1)):
        num = 0
    labels[i] = num
    num = num + 1

### FUNCTIONS ###

def probability(x,mean,var):
    stdev = math.sqrt(var)
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return float((1/(math.sqrt(2*math.pi)*stdev))*exponent)

def main():
    ######################################################################
    ########################## BAYES CLASSIFIER ##########################
    ########################## ORIGINAL DATASET ##########################
    ######################################################################

    mu = np.zeros((num_classes,num_features), dtype=float)
    var = np.zeros((num_classes,num_features), dtype=float)
    knnList = []
    heap_knn = [(item[1], item) for item in knnList]
    heapq.heapify(heap_knn)

    print("Training model...")

    # calculate mu using MLE
    for i in range(0,training_size):
        for j in range(0,num_features):
            mu[labels[i]][j] += data[i][j]
    mu = mu/(training_size/3)
    
    # calculate covariance using MLE
    for i in range(0,training_size):
        for j in range(0,num_features):
            var[labels[i]][j] += (abs(data[i][j]) - abs(mu[labels[i]][j]))**2
    var = var/(training_size/3)

    print("Training complete.")

    # metric variables
    total = 0
    correct_bayes = 0
    
    print("Testing model using Bayes Classifier...")

    # calculate probabilities
    for i in range(training_size,total_objects):
        p = np.full((num_classes), 1.0, dtype=float)
        for j in range(0,num_features):
            for k in range(0,num_classes):
                p[k] *= probability(data[i][j], mu[k][j], var[k][j])

        # Bayes classification

        decision = np.argmax(p)
            
        if (decision == labels[i]):
            correct_bayes +=1

        total+=1

    print("Testing complete.")

    bayes_metric = float(100.0*(correct_bayes/total))
    print("Bayes Classifier Success Rate: %.1f%%" % bayes_metric)
    

    ######################################################################
    ######################## K-NEAREST  NEIGHBORS ########################
    ########################## ORIGINAL DATASET ##########################
    ######################################################################

    correct_knn = 0
    test=0
    print("Testing model using K-Nearest Neighbors...")

    for i in range(training_size,total_objects):
        dist = 0.0
        p = np.full((num_classes), 0)
        for j in range(0,training_size):
            for k in range(0,num_features):
                dist += abs(abs(data[i][k]) - abs(data[j][k]))**2
            
            #kNN sort
            dist = (np.sqrt(dist))*(-1)
            heapq.heappush(heap_knn, [dist, labels[j]])

            #clean up heap, maintain heap size k
            while len(heap_knn) > kneighbors:
                heapq.heappop(heap_knn)

        #kNN classification
        
        while len(heap_knn) > 0:
            temp = heapq.heappop(heap_knn)

            p[temp[1]] += 1

        decision_knn = np.argmax(p)

        if (decision_knn == labels[i]):
            correct_knn +=1
        test += 1

        sys.stdout.write("\rTesting model... %d%% complete" % (test*100/(total_objects-training_size)))
        sys.stdout.flush()
    
    knn_metric = float(100.0*(correct_knn/test))
    print("\nK-Nearest Neighbor Classifier Success Rate: %.1f%%" % knn_metric)
    
if __name__ == '__main__':
    main()