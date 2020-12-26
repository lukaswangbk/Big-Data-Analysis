# Identifying and Counting Heavy Hitters in Data Streams
## Problem Setup
Here we implement the 1-pass data stream processing algorithm that can hopefully identify all the words appearing more than 1% of the total number of words in Shakespeare’s Works. You can download the dataset from begining page.
## Local version
```python
from numpy import linalg
import array
import numpy as np
import sklearn
from numpy import dot

f = open("./shakespeare-basket1.txt")
lines = []
line = f.readline()
while line:
    line = line.strip()
    for i in line.split(' '):
        lines.append(i)
    line = f.readline()
f.close()

f = open("./shakespeare_basket2.txt")
line = f.readline()
while line:
    line = line.strip()
    for i in line.split(' '):
        lines.append(i)
    line = f.readline()
f.close()

counter = 0
set_list = {}.fromkeys(lines).keys()
HH = []
for i in set_list:
    counter += 1
    if lines.count(i) > len(lines)/100:
        HH.append(i)
        print(HH)
        print(lines.count(i))
#    if counter % (len(lines) // 20) == 0:
#        print("finish:" + counter // (len(lines) // 20) + "/20")
```
```
output: 
['the']
1158114
['the', 'of']
726442
['the', 'of', 'and']
589730
['the', 'of', 'and', 'a']
412044
['the', 'of', 'and', 'a', 'in']
373964
['the', 'of', 'and', 'a', 'in', 'to']
489772
```
## Mapreduce
### Mapper

### Reducer

### Remark
In this 1-pass stream algorithm, it may fail to produce correct results in some scenarios.
