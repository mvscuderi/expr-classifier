from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np 
from numpy import matrix
from numpy import linalg
import torch
import math
import heapq
import sys
import time
#np.set_printoptions(threshold=sys.maxsize)

# import data from .mat file and format using PyTorch
data_import = loadmat('data.mat')
data = torch.from_numpy(data_import['face'])
data = data.permute(2, 0, 1)
data = data.numpy()
data = data.reshape(600,24*21)
data = data + 2.0 #make all positive for covariance matrix

# constants
training_size = 405
total_objects = 600
num_classes = 3
num_features = 504
kneighbors = 5
eig_size = 10

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
    ############################ LDA  DATASET ############################
    ######################################################################

    mu_class = np.zeros((3,num_features), dtype=float)
    mu_feature = np.zeros((1,num_features), dtype=float)
    mu_pp = np.zeros((3,num_features), dtype=float)
    var_pp = np.zeros((3,num_features), dtype=float)
    Sb = np.zeros((num_features,num_features), dtype=float)
    Sw = np.zeros((num_features,num_features), dtype=float)
    Sw1 = np.zeros((num_features,num_features), dtype=float)
    Sw2 = np.zeros((num_features,num_features), dtype=float)
    Sw3 = np.zeros((num_features,num_features), dtype=float)
    ident = np.identity(num_features, dtype=float)
    ident *= 0.1
    knnList = []
    heap_knn = [(item[1], item) for item in knnList]
    heapq.heapify(heap_knn)

    print("Applying LDA to dataset...")

    # calculate mu using MLE
    for i in range(0,total_objects):
        for j in range(0,num_features):
            mu_class[labels[i]][j] += data[i][j]
            mu_feature[0][j] += data[i][j]
    mu_class /= (training_size/3)
    mu_feature /= training_size
    mu_tf = np.zeros((num_features), dtype=float)

    class_1 = data[0:total_objects:3]
    class_2 = data[1:total_objects:3]
    class_3 = data[2:total_objects:3]

    for i in range(0,num_features):
        mu_tf[i] = np.mean(data[:,i])

    mu_c1 = np.zeros((num_features), dtype=float)
    mu_c2 = np.zeros((num_features), dtype=float)
    mu_c3 = np.zeros((num_features), dtype=float)
    class_1 = class_1.T
    class_2 = class_2.T
    class_3 = class_3.T

    for i in range(0,num_features):
        mu_c1[i] = np.mean(class_1[i])
    
    for i in range(0,num_features):
        mu_c2[i] = np.mean(class_2[i])

    for i in range(0,num_features):
        mu_c3[i] = np.mean(class_3[i])

    class_1 = class_1.T
    class_2 = class_2.T
    class_3 = class_3.T

    for i in range(0,int(total_objects/3)):
        Sw1 = (class_1[i].reshape(num_features,1) - mu_c1.reshape(num_features,1))*(class_1[i].reshape(num_features,1) - mu_c1.reshape(num_features,1)).T

    for i in range(0,int(total_objects/3)):
        Sw2 += (class_2[i].reshape(num_features,1) - mu_c2.reshape(num_features,1)).dot((class_2[i].reshape(num_features,1) - mu_c2.reshape(num_features,1)).T)

    for i in range(0,int(total_objects/3)):
        Sw3 += (class_3[i].reshape(num_features,1) - mu_c3.reshape(num_features,1)).dot((class_3[i].reshape(num_features,1) - mu_c3.reshape(num_features,1)).T)       
    
    Sw = num_features*(ident+ Sw1 + Sw2 + Sw3)
    
    for i in range(0,3):
        Sb += ((mu_class[i] - mu_tf).dot((mu_class[i] - mu_tf).T)) 

    M = (np.linalg.inv(Sw)).dot(Sb)

    eig_vals, eig_vecs = np.linalg.eig(M)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    val_norm = eig_vals
    val_norm/= np.sum(eig_vals)

    pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    #for pair in pairs:
        #print(pair[0])

    #print('Explained Variance')
    #for i, pair in enumerate(pairs):
        #if (i<5):
            #print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
    
    #w_matrix = np.hstack((pairs[0][1].reshape(num_features,1),pairs[1][1].reshape(num_features,1),pairs[2][1].reshape(504,1),pairs[3][1].reshape(504,1),pairs[4][1].reshape(504,1),pairs[5][1].reshape(504,1),pairs[6][1].reshape(504,1),pairs[7][1].reshape(504,1)
                #,pairs[8][1].reshape(504,1),pairs[9][1].reshape(504,1),pairs[10][1].reshape(504,1),pairs[11][1].reshape(504,1),pairs[13][1].reshape(504,1),pairs[5][1].reshape(504,1),pairs[14][1].reshape(504,1)
                #,pairs[15][1].reshape(504,1),pairs[16][1].reshape(504,1),pairs[17][1].reshape(504,1),pairs[18][1].reshape(504,1),pairs[19][1].reshape(504,1),pairs[20][1].reshape(504,1),pairs[21][1].reshape(504,1),pairs[22][1].reshape(504,1))).real
    
    eigs = np.zeros((eig_size,504))
   
    test = [pairs[0][1].reshape(504,1)]
   
    for i in range(0,eig_size):
        eigs[i] = pairs[i][1]#.reshape(504,1)
    eigs = eigs.reshape(504,eig_size)
    
    Y = data.dot(eigs)
    print("LDA complete.")
    
    # Train model
    print("Training model...")

    train_1 = Y[0:training_size:num_classes]
    train_2 = Y[1:training_size:num_classes]
    train_3 = Y[2:training_size:num_classes]

    for i in range(0,eig_size):
        mu_pp[0][i] = np.mean(train_1[:,i])
        var_pp[0][i] = np.var(train_1[:,i])
    
    for i in range(0,eig_size):
        mu_pp[1][i] = np.mean(train_2[:,i])
        var_pp[1][i] = np.var(train_2[:,i])
    
    for i in range(0,eig_size):
        mu_pp[2][i] = np.mean(train_3[:,i])
        var_pp[2][i] = np.var(train_3[:,i])

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
        for j in range(0,eig_size):
            p0 *= probability(Y[i][j], mu_pp[0][j], var_pp[0][j])
            p1 *= probability(Y[i][j], mu_pp[1][j], var_pp[1][j])
            p2 *= probability(Y[i][j], mu_pp[2][j], var_pp[2][j])

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
    ############################ LDA  DATASET ############################
    ######################################################################
    
    correct_knn = 0
    test=0
    for i in range(training_size,total_objects):
        dist = 0.0
        knn_0 = 0
        knn_1 = 0
        knn_2 = 0
        for j in range(0,training_size):
            for k in range(0,eig_size):
                dist += (abs(Y[i][k]) - abs(Y[j][k]))**2
            
            #kNN sort
            if (dist != 0.0):
                dist = (np.sqrt(abs(dist)))*(-1)
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

        decision_knn = -1   #for data validation/diagnostic purposes

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