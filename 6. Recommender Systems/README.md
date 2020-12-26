# Recommender Systems
Consider the following incomplete movie rating matrix:
USER | Movie A | Movie B | Movie C | Movie D | Movie E | Movie F
--- | --- | --- | --- | --- | --- | ---
User I | 2 | 1 | 5 | 4 | 3
User II | | 2 | | 3 | 5 | 4
User III | 5 | | 4 | 1 | 4 | 2
User IV | 2 | 3 | 4 | 5
User V | | 4 | 1 | | 3 | 2
## The predicted rating of User II on Movie C
### Item-Item Collaborative Filtering
```python
def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den

M = np.array([[2, 1, 5, 4, 3, 0],
           [0, 2, 0, 3, 5, 4],
           [5, 0, 4, 1, 4, 2],
           [2, 3, 4, 5, 0, 0],
           [0, 4, 1, 0, 3, 2]], dtype = float)
center = np.zeros(M.shape)
mean = 0
for i in range(5):
    mean = non_zero_mean(M[i])
    for j  in range(6):
        if M[i,j] != 0:
            center[i,j] = M[i,j] - mean
# print("centered M:")
# print(center)

similarity = np.zeros((5, ))
for i in range(5):
    similarity[i] = dot(center[1],center[i]) / (linalg.norm(center[1]) * linalg.norm(center[i]))
print("Similarity:")
print(similarity)

prediction = (similarity[0] * M[0][2] + similarity[2] * M[2][2]) / (similarity[0] + similarity[2]) # threshold = 0.2
print("Prediction:")
print (prediction)
```
```
output: Similarity: [0.35355339, 1., 0.2313407, 0., -0.35]
        Prediction: 4.604474207435069
```
### User-User Collaborative Filtering
```python
def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den

M = np.array([[2, 1, 5, 4, 3, 0],
           [0, 2, 0, 3, 5, 4],
           [5, 0, 4, 1, 4, 2],
           [2, 3, 4, 5, 0, 0],
           [0, 4, 1, 0, 3, 2]], dtype = float).T
center = np.zeros(M.shape)
mean = 0
for i in range(6):
    mean = non_zero_mean(M[i])
    for j  in range(5):
        if M[i,j] != 0:
            center[i,j] = M[i,j] - mean
# print("centered M:")
# print(center)

similarity = np.zeros((6, ))
for i in range(5):
    similarity[i] = dot(center[2],center[i]) / (linalg.norm(center[2]) * linalg.norm(center[i]))
print("Similarity:")
print(similarity)

prediction = (similarity[3] * M[3][1] + similarity[4] * M[4][1]) / (similarity[3] + similarity[4]) # threshold = 0.08
print("Prediction:")
print(prediction)
```
```
output: Similarity: [-0.13608276, -0.85715939, 1., 0.09860133, 0.17588162, 0.]
        Prediction: 4.281548594137618
```

## Matrix Factorization 
techniques are effective to discover the latent features
underlying the interactions between users and items. A matrix factorization example and its
Python code are provided in the blog of Ref [1]. Please read the blog in [1] to understand the
python code and then use it to predict the rating of User II on Movie C. Compare the result with
the ones you obtained in part (a).
```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    _Q, _P = Q, P
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * _Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * _P[i][k] - beta * Q[k][j])
        eR = dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        print("loss: ", e)
        if e < 0.001:
            break
    return P, Q.T

R = [
     [2, 1, 5, 4, 3, 0],
     [0, 2, 0, 3, 5, 4],
     [5, 0, 4, 1, 4, 2],
     [2, 3, 4, 5, 0, 0],
     [0, 4, 1, 0, 3, 2],
    ]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 5

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)
```