# Community Detection in Online Social Networks
###### *Remark: If the formular cannot showed properly on chrome, please add [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)
## Problem Setup
Community detection has drawn lots of attention these years in machine learning field. With the popularity of online social networks, such as Facebook, Twitter, and Sina Weibo, we can get many valuable datasets to develop algorithms. In this project, we have implement a community detection algorithm for the online social networks. Basically, people belong to the same community when they have high similarity (i.e. sharing many common interests). There are two types of relationship in online social networks. One is symmetric, which means when Alice is Bob’s friend, Bob will also be Alice’s friend; the other is asymmetric, which means Bob may not be Alice’s friend even Alice is Bob’s friend. In the second case, there are two roles in the relationship: follower and followee (like the case when using Twitter and Weibo). When Alice follows Bob, Alice is the follower and Bob is the followee.\
\
To detect communities, we need to calculate the similarity between any pair of users. In this project, similarity is measured by the number of common followees divided by the total
number of the two users’ followees for the given pair of users. The following figure illustrates the process. The following is the formal definition of similarity, where $out(A)$ is the set of all the followees of A. If $|out(A) \cup out(B)|>0$, we define
$$Similarity(A,B) = \frac{|out(A) \cap out(B)|}{|out(A) \cup out(B)|}$$
where |S| means the cardinality of S.\
(**Note** : if $|out(A) \cup out(B)|=0$, set the similarity to be 0.)\
\
![](pic/community%20detection.jpg)\
\
The set of followees of A is {B, C, E} and set of followees of B is {A, C, E}. There are 2 common followees between A and B (i.e. C and E), and the number of the union of their followees is 4. The similarity between A and B is therefore 2/4 = 0.5.\
\
We will use three datasets with different sizes, generated from FaceBook, Twitter, and Google+ users’ relationship. The smaller dataset contains 4K users. The medium one contains around 80K users, and the large one contains 100K users. Each user is represented by its unique ID number. The download links of three datasets are listed in the the begining page. The small dataset is used to facilitate initial debugging and testing.\
\
The format of the data file of the above example is as follows:\
A$\quad$B\
A$\quad$D\
B$\quad$A\
C$\quad$A\
C$\quad$B\
C$\quad$E\
E$\quad$A\
E$\quad$B\
E$\quad$C
## Maximal Common Followees Recommendation
Recommend the person with the maximal number of common followees in the medium scale dataset. If multiple people share the same number, randomly pick one of them.
### MR job 1: generate user pairs with common followee
#### Mapper 1
```python
# Mapper.py

#!/usr/bin/env python3

import sys

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    followee, follower = line.split()
    sys.stdout.write("%s\t%s\n" % follower, followee)
```
#### Reducer 1
### MR job 2: count the number of each kind of user pairs (same as wordcount)
#### Mapper 1
#### Reducer 1
## Top-K Similar User Recommendation
Find the top K (K=3) most similar people of EVERY user for the medium scale dataset. If multiple people have the same similarity, randomly pick three of them
## Common Followees for Similar Users
Besides the number of similar users for a target user, say A, we also want to know the IDs of the common followees shared between A and its similar
users. In our example, for User A, if B is the similar user (TopK in Q1.a) to A, the desirable
output should be:
<center>
A: B, {C, E}, check_sum for the IDs of the common_followees
</center>

## Composite Key and Secondary Sorting Optimization
Find the TOP 3 (=K) most similar people and the list of common followees for each user in the large scale dataset. Here we use the [composite key design pattern](http://tutorials.techmytalk.com/2014/11/14/mapreduce-composite-key-operation-part2/) and [secondary sorting techniques](http://codingjunkie.net/secondary-sort/) to reduce memory in order to handle large dataset.
