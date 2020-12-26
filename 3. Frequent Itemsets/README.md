# Frequent Itemsets
## Problem Setup
In this problem, we use the Yelp-review dataset. The dataset contains many user reviews, which are extracted from a website (Yelp) that publishes crowd-sourced reviews. The original data set has been pre-processed as follows:
- Apply a sliding window of 40 words on each review. All the 40 words in one window make up a basket.
- Delete duplicate words in one basket, then filter out some common words.
- You can download the pre-processed data from the begining page.
- Each line of this input is a space separated list of words which corresponds to one basket.

The threshold for a frequent pair is defined as s=0.005. The frequency of a pair = *Occurrence of pair (i, j)* / *Total number of baskets*. We output a frequent pair and its corresponding count.

## A-priori+SON
### A-priori
Implement the A-Priori algorithm to find frequent pairs on a single machine.
```python
import collections
import itertools
import sys
import os
from time import time

# start = time()  

path = './freq_singleton.txt'
if os.path.exists(path):
    os.remove(path)  
else:
    print('no such file')

# reading of the txt file   
f = open("./yelp_review")
w = open('./freq_singleton.txt','a')
s = 0.005

# A-Priori Algorithm first pass
# counter collection for the frequencies all items
counts = collections.Counter()
# dictionary with each item and its frequency
freq_singleton = {}
baskets = 0

line = f.readline()
while line:
    line = line.strip()
    if line:
        line = line.split()
        unique_row_items = set([word.lower().strip() for word in line])
        # for each unique item increase its frequency
        for unique_item in unique_row_items:
            counts[unique_item] += 1
        baskets += 1
    line = f.readline()

# for each item whose frequency is bigger than the support's value, put it in frequency dictionary  
for unique_item in counts:
    if (counts[unique_item]) >= baskets*s:
        freq_singleton[unique_item] = counts[unique_item]
print(len(freq_singleton))
for i in freq_singleton:
    w.write(str(i)+" ")
print(baskets)
w.close() 
f.close()


# A-Priori Algorithm second pass

path = './freq_pair.txt'
if os.path.exists(path):
    os.remove(path)  
else:
    print('no such file')
    
f = open("./yelp_review")
w = open('./freq_pair.txt','a')

k = 1
         
counts = collections.Counter()
# dictionary with each item and its frequency
freq = {}
line = f.readline()

start = time()
while line:
    line = line.strip()
    if line:
        line = line.split()
        if k/100000 == k//100000:
            end = time() - start
            print("line:{}|time:{:,.3f}s:".format(k,end))
            start = time()
        k+=1
        unique_row_items = list(set([word.lower().strip() for word in line]))
        # creation of item's 2 combinations
        for i in range(len(unique_row_items)-1):
            if unique_row_items[i] in freq_singleton:
                for j in range(i+1, len(unique_row_items)):
                    if unique_row_items[j] in freq_singleton:
                        pair = (unique_row_items[i],unique_row_items[j])
                        counts[tuple(sorted(pair))] += 1
    line = f.readline()
    
# for each item whose frequency is bigger than the support's value, put it in frequency dictionary
for unique_item in counts:
    if (counts[unique_item]) >= baskets*s:
        freq[unique_item] = counts[unique_item]

if len(freq) > 40:
    output = sorted(freq.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:40]
    print(output)
for i in freq:
    w.write(str(i)+":"+str(freq[i])+"\n")
w.close()    
f.close()

# end = time() - start 

# print("time = {:,.3f}s".format(end))
```
### SON
#### MR job 1
First MapReduce job use A-priori algorithm to find the candidate pairs, which are frequent in at least one input file.
##### Mapper
```python
#!/usr/bin/env python

import collections
import itertools
import sys

s = 0.005

# counter collection for the frequencies all items
counts = collections.Counter()

# dictionary with each item and its frequency
freq_singleton = {}
baskets = 0

# read file in memory
lines = []

for line in sys.stdin:
    line = line.strip()
    if line:
        lines.append(line)
        line = line.split()
        unique_row_items = set([word.lower().strip() for word in line])
        # for each unique item increase its frequency
        for unique_item in unique_row_items:
            counts[unique_item] += 1
        baskets += 1
        
# for each item whose frequency is bigger than the support's value, put it in frequency dictionary  
for unique_item in counts:
    if (counts[unique_item]) >= baskets*s:
        freq_singleton[unique_item] = counts[unique_item]
        
# A-Priori Algorithm second pass
counts = collections.Counter()

# dictionary with each item and its frequency
for line in lines:
    line = line.strip()
    if line:
        line = line.split()
        unique_row_items = list(set([word.lower().strip() for word in line]))
        # creation of item's 2 combinations
        for i in range(len(unique_row_items)-1):
            if unique_row_items[i] in freq_singleton:
                for j in range(i+1, len(unique_row_items)):
                    if unique_row_items[j] in freq_singleton:
                        pair = (unique_row_items[i],unique_row_items[j])
                        counts[tuple(sorted(pair))] += 1

# for each item whose frequency is bigger than the support's value, put it in frequency dictionary
for unique_item in counts:
    if (counts[unique_item]) >= baskets*s:
        print '%s\t%s' % (str(unique_item),str(counts[unique_item]))
```
##### Reducer
```python
#!/usr/bin/env python

from operator import itemgetter
import sys

current_word = None
for line in sys.stdin:
    line = line.strip()
    if line:
        word = line.split('\t')
        if current_word == word[0]:
            continue
        else:
            if current_word:
                print '%s' % (current_word)
                current_word = word[0]
            current_word = word[0]
# do not forget to output the last word if needed!
if current_word == word[0]:
    print '%s' % (current_word)
```
##### Merge file
Create a program to merge all the outputs from first MapReduce job and remove duplicated outputs
```python
#!/usr/bin/env python
import sys
import os
import os.path

# read file in memory
filedir = './output'
filenames = os.listdir(filedir)
f = open('MR_job_1_output.txt','w')
for filename in filenames:
    filepath = filedir+'/'+filename
    for line in open(filepath):
        f.writelines(line)
    f.write('\n')
f.close()
```
#### MR job 2
Second MapReduce job counts only the candidate frequent pairs
##### Mapper
```python 
#!/usr/bin/env python

import collections
import sys
import os
import re

# counter collection for the frequencies all items
counts = collections.Counter()

# load freqent pair
f = open("./MR_job_1_output.txt")
line = f.readline()
while line:
    line = line.strip()
    if line:
        line = re.sub(r"[\(\)\"\' ']","",line)
        line = line.split(',')
        counts[tuple(sorted(line))] = 0
    line = f.readline()

for line in sys.stdin:
    line = line.strip()
    if line:
        line = line.split()
        unique_row_items = set([word.lower().strip() for word in line])
        # for each unique item increase its frequency
        for i in counts:
            if set(i).issubset(unique_row_items):
                counts[i] += 1
for unique_item in counts:
    print '%s\t%s' % (str(unique_item),str(counts[unique_item]))
```
##### Reducer
```python
#!/usr/bin/env python

from operator import itemgetter
import sys

s = 0.005
baskets = 4984299

current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()
    if line:
        word, count = line.split('\t', 1)
        try:
            count = int(count)
        except ValueError:
            # count was not a number, so silently
            # ignore/discard this line
            continue
        # this IF-switch only works because Hadoop sorts map output
        # by key (here: word) before it is passed to the reducer
        if current_word == word:
            current_count += count
        else:
            if current_word and current_count >= s*baskets:
                print '%s\t%s' % (current_word, current_count)
            current_count = count
            current_word = word
            
# do not forget to output the last word if needed!
if current_word == word and current_count >= s*baskets:
    print '%s\t%s' % (current_word, current_count)
```
#### Shell script
```Bash
#！/bin/sh
hadoop jar /usr/hdp/2.4.2.0-258/hadoop-mapreduce/hadoop-streaming.jar -D mapred.map.tasks=10 -D mapred.reduce.tasks=10 -file /home/1155107949/IERG4300/HW2/MR_job_1/mapper.py -mapper mapper.py -file /home/1155107949/IERG4300/HW2/MR_job_1/reducer.py -reducer reducer.py -input input/yelp_review -output MR_job_1
hadoop dfs -get MR_job_1/part-* /home/1155107949/IERG4300/HW2/MR_job_1/output/
python /home/1155107949/IERG4300/HW2/MR_job_1/file_merge.py
cp /home/1155107949/IERG4300/HW2/MR_job_1/MR_job_1_output.txt IERG4300/HW2/MR_job_2/
hadoop jar /usr/hdp/2.4.2.0-258/hadoop-mapreduce/hadoop-streaming.jar -D mapred.map.tasks=10 -D mapred.reduce.tasks=10 -file /home/1155107949/IERG4300/HW2/MR_job_2/MR_job_1_output.txt -file /home/1155107949/IERG4300/HW2/MR_job_2/mapper.py -mapper mapper.py -file IERG4300/HW2/MR_job_2/reducer.py -reducer reducer.py -input input/yelp_review -output MR_job_2
sort -n -k 2 -t $'\t' /home/1155107949/IERG4300/HW2/MR_job_2/MR_job_2_output.txt -o /home/1155107949/IERG4300/HW2/MR_job_2/MR_job_2_output.txt
```
## PCY
Use the PCY algorithm to filter the candidate pairs in the
SON algorithm with the following Hash Function:
$$HashFunction=hash( word\_1 + word\_2) \% 100000$$

## Minhash/ Locality-Sensitive Hashing (LSH)
The following is a matrix representing three sets, X, Y, and Z, and a universe of five elements a through e.
Row | X | Y | Z 
--- | --- | --- | ---
a | 1 | 0 | 1
b | 1 | 1 | 0 
c | 0 | 1 | 1
d | 1 | 0 | 0
e | 0 | 1 | 1

**Jaccard similarities:**\
Sim(X, Y) = 1/5 \
Sim(X, Z) = 1/5 \
Sim(Y, Z) = 2/4 = 1/2
### Minhash
Suppose we create Minhash signatures of length 5 for each of the three sets X, Y, and Z. The signatures are based on the five cyclic permutations of the rows. That is, the first permutation uses order **abcde**, the second uses **bcdea**, the third uses **cdeab**, the fourth **deabc**, and the fifth **eabcd**. The signature matrix is below.
Perm | X | Y | Z 
--- | --- | --- | ---
1 | 1 | 2 | 1
2 | 2 | 1 | 1 
3 | 1 | 2 | 2
4 | 2 | 1 | 1
5 | 1 | 1 | 2

**Jaccard similarities:**\
Sim(X, Y) = 1/5\
Sim(X, Z) = 1/5\
Sim(Y, Z) = 3/5

### General Formular
Consider the use of the Minhash/Locality-Sensitive Hashing (LSH) scheme to find similar items based on their corresponding columns in the Minhash signature matrix M . Let n = b · r be the number of rows in M and the n rows of M are divided into b bands of r rows each. For each band, we hash its portion of each column to a hash table with k buckets where k is set to be large enough so that the effect of hash collision is negligible. A pair of items is considered to be a similar-pair candidate if their corresponding columns are hashed to the same bucket in one or more bands. For 2 items C1 and C2 have similarity s , namely, their corresponding signature columns in M actually agree on s fraction of the rows in M . The probability that C1 and C2 will NOT be identified as a similar-pair candidate is **$(1-s^{r})^{b}$**.
### Parameter Design for Minhash/ Locality-Sensitive Hashing (LSH)
Let r be the number of rows within each band and B be the total number of bands within the Minhash signature matrix M. We want to design the system so that:
- For any pair of items with similarity greater than or equal to T1, the probability that they will be correctly identified as a similar-pair candidate should be at least P1.
- For any pair of items with similarity below T2, the probability that they will be mistakenly identified as a similar-pair candidate should be no more than P2.\

For T1=0.85, T2=0.5, P1=0.99 and P2=0.01, derive a single pair of values for (r, B) so that the aforementioned accuracy/error requirements would be satisfied.
```python 
from sympy import Pow
import numpy as np

for i in range(100):
    for j in range(100):
        f1 = 1-Pow((1-Pow(0.85,i)),j)
        f2 = 1-Pow((1-Pow(0.5,i)),j)
        if f1 >= 0.99 and f2 < 0.01:
            print('r:{0}, s:{1}, f1:{2}, f2:{3}'.format(i,j,f1,f2))
            break
    if f1 >= 0.99 and f2 < 0.01:
        break
```