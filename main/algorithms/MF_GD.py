import pandas as pd
import numpy

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''

    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:
            break

    return P, Q.T

R = [
     [1, 0, 5, 2, 4, 2, 1, 3, 4, 3],
     [3, 4, 5, 2, 1, 0, 5, 4, 2, 3],
     [4, 3, 1, 1, 5, 2, 0, 4, 5, 3],
     [2, 5, 0, 3, 3, 4, 2, 1, 4, 5],
     [5, 2, 3, 0, 4, 3, 2, 1, 1, 2],
     [3, 2, 1, 4, 0, 5, 5, 1, 2, 3],
     [1, 1, 4, 2, 5, 0, 3, 5, 4, 3],
     [4, 2, 3, 5, 5, 1, 0, 4, 2, 1],
     [3, 5, 4, 1, 2, 0, 1, 3, 5, 4],
     [2, 3, 1, 4, 5, 2, 3, 1, 0, 2],
    ]

R = numpy.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

 
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)

nR = numpy.dot(nP, nQ.T)

print(nR)