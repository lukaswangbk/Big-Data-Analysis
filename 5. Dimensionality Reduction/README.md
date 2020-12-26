# Dimentionality Reduction
## Singular Value Decomposition (SVD)
The matrix below is a document word matrix which shows the number × of times a particular word occurs in some given documents. Here, each row represents a document which is characterized by the number of times each of the 7 words appeared in the document.
FILE | energy | colorful | speed | elegant | spin | direction | artistic 
--- | --- | --- | --- | --- | --- | --- | ---
Doc1 | 2 | 5 | 0 | 6 | 0 | 3 | 5
Doc2 | 1 | 4 | 9 | 0 | 0 | 3 | 0
Doc3 | 5 | 0 | 5 | 0 | 4 | 1 | 1
Doc4 | 0 | 8 | 0 | 4 | 8 | 1 | 0
Doc5 | 1 | 9 | 6 | 3 | 7 | 8 | 9
Doc6 | 10 | 2 | 3 | 0 | 5 | 2 | 5

Originally, each document is located in a 7-D space. Using SVD, this set of documents can be approximately embedded into a 2-D space instead. 
```python
from numpy import linalg
import array
import numpy as np
import sklearn
from numpy import dot

A=np.mat([[2, 5, 0, 6, 0, 3, 5],
         [1, 4, 9, 0, 0, 3, 0],
         [5, 0, 5, 0, 4, 1, 1],
         [0, 8, 0, 4, 8, 1, 0],
         [1, 9, 6, 3, 7, 8, 9],
         [10, 2, 3, 0, 5, 2, 5]])
print("Input")
print(A)
U, Sigma, VT=linalg.svd(A)  
print("U", U)
print("Sigma", Sigma)
print("VT", VT)
Sig = np.mat([[Sigma[0],0], [0,Sigma[1]]]) # Change to 2-D
print("Sig", Sig)
print("U'", U[:,:2])
print("VT'", VT[:2,:])
B = U[:,:2] * Sig * VT[:2,:]
print("Output")
print(B)
```

For a new Doc7, the number of times each of the words appeared is [0 5 4 0 3 2 2]. What is the representation of Doc7 in the “concept” space?

```python
data = [0, 5, 4, 0, 3, 2, 2]
concept = data * (VT[:2,:]).T
print("concept space")
print(concept)
```

Compute the cosine similarities between Doc3 and Doc6 based on their vectors in the “word” space and the “concept” space.

```python
concept_doc3 = A[2] * (VT[:2,:]).T
concept_doc3 = np.squeeze(np.asarray(concept_doc3))

concept_doc6 = A[5] * (VT[:2,:]).T
concept_doc6 = np.squeeze(np.asarray(concept_doc6))

concept_cos_similarity = dot(concept_doc3,concept_doc6) / (linalg.norm(concept_doc3) * linalg.norm(concept_doc6))
print("concept space cosine similarity", concept_cos_similarity)

word_doc3 = np.squeeze(np.asarray(A[2]))
word_doc6 = np.squeeze(np.asarray(A[5]))

word_cos_similarity = dot(word_doc3,word_doc6) / (linalg.norm(word_doc3) * linalg.norm(word_doc6))
print("word space cosine similarity", word_cos_similarity)
```
## K-means with PCA
### Perform PCA on dataset
```python
  
from __future__ import print_function

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import *

def pca(x):
    """Performs principal component on x, a matrix with observations in the rows.
    Returns the projection matrix (the eigenvectors of x^T x, ordered with largest eigenvectors first) and the eigenvalues (ordered from largest to smallest).
    """
    
    x = (x - x.mean(axis = 0)) # Subtract the mean of column i from column i, in order to center the matrix.
    
    num_observations, num_dimensions = x.shape
    
    # Often, we have a large number of dimensions (say, 10,000) but a relatively small number of observations (say, 75). In this case, instead of directly computing the eigenvectors of x^T x (a 10,000 x 10,000 matrix), it's more efficient to compute the eigenvectors of x x^T and translate these into the eigenvectors of x^T x by using the transpose trick. 
    # The transpose trick says that if v is an eigenvector of M^T M, then Mv is an eigenvector of MM^T.
   
    if num_dimensions > 25:
        eigenvalues, eigenvectors = linalg.eigh(dot(x, x.T))
        v = (dot(x.T, eigenvectors).T)[::-1] # Unscaled, but the relative order is still correct.
        s = sqrt(eigenvalues)[::-1] # Unscaled, but the relative order is still correct.
    else:
        u, s, v = linalg.svd(x, full_matrices = False)
        
    return v, s

def main():

    dataset_train = np.zeros((50000,784))
    label_train = np.zeros(50000)
    dataset_test = np.zeros((10000,784))
    label_test = np.zeros(10000)

    train_name = './dataset/mnist_train.txt'
    test_name = './dataset/mnist_test.txt'

    fname = 'pca_test.txt'
    
    with open(fname,'w') as f:
        count = 0
        with open(test_name) as te:
            lines = [line.rstrip('\n') for line in open(test_name)]
            for line in lines:
                line = line.strip()
                header, line = line.split(':')
                header, digit = header.split()
                pixels = line.split()
                if(count < 10000):
                    for i in range(0,784):
                            dataset_test[count][i] = int(pixels[i])
                    label_test[count] = int(digit)
                count += 1

        pcaModel = PCA()
        pcaModel.fit(dataset_test)
        eigenValues = pcaModel.explained_variance_
        N_comp = 25

        result = PCA(n_components = N_comp)

        dataset_test = result.fit_transform(dataset_test)

        for i in range(0,np.shape(label_test)[0]):
            out_str = "digit "
            out_str += str(label_test[i]) + ':'
            for k in range(0,N_comp):
                out_str += str(dataset_test[i][k]) + " "
            out_str = out_str[:-1] + '\n'
            f.write(out_str)

    fname = 'pca_train.txt'
    with open(fname,'w') as f:
        count = 0
        # Load the dataset
        with open(train_name) as tr:
            lines = [line.rstrip('\n') for line in open(train_name)]
            for line in lines:
                line = line.strip()
                header, line = line.split(':')
                header, digit = header.split()
                pixels = line.split()
                if(count < 50000):
                    for i in range(0,784):
                            dataset_train[count][i] = int(pixels[i])
                    label_train[count] = int(digit)
                    count += 1
        
        pcaModel = PCA()
        pcaModel.fit(dataset_train)
        eigenValues = pcaModel.explained_variance_
        N_comp = 25

        result = PCA(n_components = N_comp)

        dataset_train = result.fit_transform(dataset_train)

        for i in range(0,np.shape(label_train)[0]):
            out_str = "digit "
            out_str += str(label_train[i]) + ':'
            for k in range(0,N_comp):
                out_str += str(dataset_train[i][k]) + " "
            out_str = out_str[:-1] + '\n'
            f.write(out_str)

if __name__ == '__main__':
    main()
```
### Mapper
```python
#!/usr/bin/env python

import sys
import os
import random as rnd
import numpy as np

fname = 'centroid.txt'

centroid = np.zeros((10,5,5))

def distance(pt1,cen_num):
    dist = 0
    for r in range(0, 5):
        for c in range(0, 5):
            dist += (float(pt1[r*5+c]) - centroid[cen_num][r][c]) * (float(pt1[r*5+c]) - centroid[cen_num][r][c])
    return dist

if os.path.exists(fname):
    if os.path.getsize(fname) == 0:
        with open(fname,"a") as f:
            for o in range(0,10):
                w_str = 'Centroid ' + str(o) + ':'
                for k in range(0, 5*5):
                    w_str += str(rnd.randint(0,255)) + ' '
                w_str = w_str[:-1]
                w_str += ',0\n'
                f.write(w_str)
        f.close()
        exit()


with open(fname) as f:
    lines = [line.rstrip('\n') for line in open(fname)]
    for line in lines:
        line = line.strip()
        line, count = line.split(',')
        header, cents = line.split(':')
        header, index = header.split()
        cents = cents.split()
        for r in range(0,5):
            for c in range(0,5):
                tot = r * 5 + c
                if(tot % 2 == 0):
                    centroid[int(index)][r][c] = float(cents[tot])
                else:
                    centroid[int(index)][r][c] = float(cents[tot])

for line in sys.stdin:
    line = line.strip()
    header, line = line.split(':')
    header, digit = header.split()
    pixels = line.split()

    close_distance = distance(pixels,0)
    close_index = 0
    for i in range(1,10):
        new_dist = distance(pixels,i)
        if(close_distance > new_dist):
            close_distance = new_dist
            close_index = i

    str_pass = ""
    for pixel in pixels:
        str_pass = str_pass + str(pixel) + " "
    str_pass = str_pass[:-1]

    print '%s\t%s' % (str(close_index), str_pass)
```
### Reducer
```python
#!/usr/bin/env python

import sys
import numpy as np

current_index = None

centroid = np.zeros(5*5)
num_member = 0

for line in sys.stdin:
    index, pixels = line.strip().split('\t')
    pixels = pixels.split()

    if current_index:
    	if(index == current_index):
    		for r in range(0,5):
    			for c in range(0,5):
    				centroid[r*5+c] += float(pixels[r*5+c])
    		num_member += 1
    	else:
    		w_str = 'Centroid ' + current_index + ':'
    		for i in range(0, 5*5):
    			w_str += str(centroid[i]/num_member) + " "
    		w_str = w_str[:-1]
    		w_str += "," + str(num_member) + "\n"
    		print(w_str)
    		centroid = np.zeros(5*5)
    		num_member = 0
    current_index = index

if(current_index>1):
	w_str = 'Centroid ' + str(current_index) + ':'
	for i in range(0, 5*5):
		w_str += str(centroid[i]/num_member) + " "
	w_str = w_str[:-1]
	w_str += "," + str(num_member)
	print(w_str)
```
### Shell Script
```Bash
#!/bin/bash
chmod +x mapper_kmeans.py mapper_kmeans.py centroid.txt centroid_new.txt centroid_all.txt

for i in `seq 1 10`;
do
	hdfs dfs -rm -r /user/ly116/output_pca

	hdfs dfs -mkdir /user/ly116/pca_train
	hdfs dfs -put ./dataset/pca_train.txt /user/ly116/pca_train

	hadoop jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar \
	-D mapreduce.map.memory.mb=2048 \
	-D mapreduce.reduce.memory.mb=1024 \
	-D mapred.map.tasks=20 \
	-D mapred.reduce.tasks=10 \
	-input /user/ly116/pca_train/* \
	-output output_pca \
	-file mapper_kmeans.py -mapper mapper_kmeans.py \
	-file reducer_kmeans.py -reducer reducer_kmeans.py \
	-file centroid.txt

	hdfs dfs -cat /user/ly116/output_pca/* > centroid_new.txt

	cat centroid_new.txt > centroid.txt
	cat centroid_new.txt >> centroid_all.txt

	hdfs dfs -rm -r /user/ly116/output_pca
done
```