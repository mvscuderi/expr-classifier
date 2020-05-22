from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np 
from numpy.linalg import eig
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

#do not change; for PCA. used to choose to do PCA for 95% or 90% variance later on, must be changed line 102
var_90_cnt = 0  
var_95_cnt = 0

# constants *can be edited to change test/training size etc.*
training_size = 99
total_objects = 600
num_classes = 3
num_features = 504
kneighbors = 5
pca_size = var_95_cnt
pca_size = var_90_cnt

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
    x = abs(x)
    #print(stdev)
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return float((1/(math.sqrt(2*math.pi)*stdev))*exponent)

def main():
    ######################################################################
    ########################## BAYES CLASSIFIER ##########################
    ############################ PCA  DATASET ############################
    ######################################################################

    mu_feature = np.zeros((num_features), dtype=float)
    var_feature = np.zeros((num_features), dtype=float)
    mu_pp = np.zeros((3,num_features), dtype=float)
    var_pp = np.zeros((3,num_features), dtype=float)
    knnList = []
    heap_knn = [(item[1], item) for item in knnList]
    heapq.heapify(heap_knn)

    print("Applying PCA to dataset...")

    pca_data = data
    M = np.mean(pca_data.T, axis=1)
    for i in range(0,num_features):
        mu_feature[i] = np.mean(pca_data[:,i])
        var_feature[i] = np.var(pca_data[:,i])
    
    for i in range(0,600):
        for j in range(0,num_features):
            pca_data[i][j] = (pca_data[i][j] - mu_feature[j])/var_feature[j]

    
    V = np.cov(pca_data.T)
    values, vectors = eig(V)
    
    pca_size = i
    values_norm = values
    values_norm /= np.sum(values)
    values_norm = values_norm.real
    var_90 = 0.0
    var_95 = 0.0
    var_90_cnt = 0
    var_95_cnt = 0
    for i in range(len(values_norm)):
        if (var_90 < 0.90):
            var_90 += values[i]
            var_90_cnt += 1
        if (var_95 < 0.95):
            var_95 += values[i]
            var_95_cnt += 1
        if (var_95 > 0.950 and var_90 > 0.90):
            break

    print(var_90_cnt)
    print(var_95_cnt)
        
    pca_size = var_95_cnt

    #print(values[0:10])
    #print(values)
    #print(values.shape)
    vectors = vectors[0:pca_size]
    
    P = np.dot(pca_data, vectors.T)
    #eigen = sorted(eigen, key=lambda tup: abs(tup[1]))
    
    print("PCA complete.")

    size = np.arange(1,22)
    varr = np.zeros(61,dtype=float)
    for i in range(0,21):
        varr[i] += values_norm[i]
    
    # Train model
    print("Training model...")

    train_1 = P[0:training_size:num_classes]
    train_2 = P[1:training_size:num_classes]
    train_3 = P[2:training_size:num_classes]

    for i in range(0,pca_size):
        mu_pp[0][i] = np.mean(train_1[:,i]).real
        var_pp[0][i] = np.var(train_1[:,i]).real
    
    for i in range(0,pca_size):
        mu_pp[1][i] = np.mean(train_2[:,i]).real
        var_pp[1][i] = np.var(train_2[:,i]).real
    
    for i in range(0,pca_size):
        mu_pp[2][i] = np.mean(train_3[:,i]).real
        var_pp[2][i] = np.var(train_3[:,i]).real
    
    print("Training complete.")
    
    # metric variables
    total = 0
    correct_bayes = 0

    print("Testing model using Bayes Classifier...")
    
    # calculate probabilities
    for i in range(training_size,total_objects):
        p0 = 1.0
        p1 = 1.0
        p2 = 1.0
        for j in range(0,pca_size):
            p0 *= probability(P[i][j], mu_pp[0][j], var_pp[0][j])
            p1 *= probability(P[i][j], mu_pp[1][j], var_pp[1][j])
            p2 *= probability(P[i][j], mu_pp[2][j], var_pp[2][j])

        # Bayes classification
        decision = -1 # for diagnostic

        if ((p0>p1) and (p0>p2)):
            decision=0
        elif ((p1>p0) and (p1>p2)):
            decision=1
        elif ((p2>p1) and (p2>p0)):
            decision=2
            
        if (decision == labels[i]):
            correct_bayes +=1

        total+=1

    print("Testing complete.")
    bayes_metric = float(100.0*(correct_bayes/total))
    print("\nBayes Classifier Success Rate: %.1f%%" % bayes_metric)
    
    
    ######################################################################
    ######################## K-NEAREST  NEIGHBORS ########################
    ############################ PCA  DATASET ############################
    ######################################################################
    
    correct_knn = 0
    test=0
    for i in range(training_size,total_objects):
        dist = 0.0
        knn_0 = 0
        knn_1 = 0
        knn_2 = 0
        for j in range(0,training_size):
            for k in range(0,num_features):
                dist += abs(abs(pca_data[i][k]) - abs(pca_data[j][k]))**2
            
            #kNN sort
            dist = (np.sqrt(dist))*(-1)
            heapq.heappush(heap_knn, [dist, labels[j]])

            #clean up heap, maintain heap size k
            while len(heap_knn) > kneighbors:
                heapq.heappop(heap_knn)

        #kNN classification
        
        while len(heap_knn) > 0:
            temp = heapq.heappop(heap_knn)

            if (temp[1]==0):
                knn_0 += 1
            elif (temp[1]==1):
                knn_1 += 1
            elif (temp[1]==2):
                knn_2 += 1
        decision_knn = -1
        if ((knn_0 >= knn_1) and (knn_0 >= knn_2)):
            decision_knn = 0
        if ((knn_1 >= knn_0) and (knn_1 >= knn_2)):
            decision_knn = 1
        if ((knn_2 >= knn_1) and (knn_2 >= knn_0)):
            decision_knn = 2

        if (decision_knn == labels[i]):
            correct_knn +=1
        test +=1

        sys.stdout.write("\rTesting model... %d%% complete" % (test*100/(total_objects-training_size)))
        sys.stdout.flush()
    
    knn_metric = float(100.0*(correct_knn/test))
    print("\nK-Nearest Neighbor Classifier Success Rate: %.1f%%" % knn_metric)
    
    
if __name__ == '__main__':
    main()