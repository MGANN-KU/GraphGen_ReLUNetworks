# -*- coding: utf-8 -*-
"""
Created on Sat September 13,2025 at 07:14:00

@author: Ghafoor
"""

import numpy as np
#Deletion in Graph edit distance

# parameters
eps = 1e-7
C = 1e5
B = 1e3

# Input
d = 3
m = 5

# one- dimentional label matrix 6 vertices
L = [2, 3, 5, 4, 2,4]
#Example for a graph with 6 vertices
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    ] 
n = len(A)
# First layer is input layer, Given as
x = [4, 3, 7, 6, 3, 2, 5, 1, 2]
X = x  
print("Input:")
print("d:", d)
print("m:", m)
print("x:", X)
###################################################################################       
###################################################################################
#                    Padding d number of B at the end of L and 
#                       d number of rows and columns in A
###################################################################################
###################################################################################
# Create U by padding d instances of B to L
U = L + [B] * ( d)
#print("Padded list U:", U)
#
def pad_adjacency_matrix(A, d, B):
    n = len(A)  # Original size of the matrix
    # First pad each existing row with d B's
    for row in A:
        row.extend([B] * d)
    # Now add d new rows, each of size n + d filled with B
    for _ in range(d):
        A.append([B] * (n + d))
    return A

# Call the function and assign the result
V = pad_adjacency_matrix(A, d, B)

# Print the padded matrix
# print("Padded Adjacency Matrix V:")
# for row in V:
#     print(row)
#### Flatten 2D list to 1D
V1 = [item for row in V for item in row]
# print(V1)
#####################################################################
# To construct weight matrix for second layer/first hidden layer is L1=W1*X+B1
W1 = []

## (eta1, eta2 to solve delta(x_j, x_{j+d}))
for j in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for k in range(1, 3*d+1):
          w = 0
          if j == k:
                w = 1/eps
          if j+d == k:
                  w=-1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## (eta3, eta4 to solve delta(x_j, x_{j+d}))
for j in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for k in range(1, 3*d+1):
          w = 0
          if j == k:
                w = -1/eps
          if j+d == k:
                  w=1/eps
          temp_row.append(w)
      W1.append(temp_row)
####
## (alpha1, alpha2 to solve delta(x_j, x_k)
for j in range(1, d+1):
  for k in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for i in range(1, 3*d+1):
          w = 0
          if j == i and j!=k:
                w = 1/eps
          if k == i and j!=k:
                w = -1/eps      
          temp_row.append(w)
      W1.append(temp_row)
#####
## (beta1, beta2 to solve delta(x_j, x_k)
for j in range(1, d+1):
  for k in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for i in range(1, 3*d+1):
          w = 0
          if j == i and j!=k:
                w = -1/eps
          if k == i and j!=k:
                w = 1/eps      
          temp_row.append(w)
      W1.append(temp_row)
#####
## (alpha'1, alpha'2 to solve delta(x_j+d, x_k+d)
for j in range(d+1, 2*d+1):
  for k in range(d+1, 2*d+1):  
    for q in range(2): 
      temp_row = []
      for i in range(1, 3*d+1):
          w = 0
          if j == i and j!=k:
                w = 1/eps
          if k == i and j!=k:
                w = -1/eps      
          temp_row.append(w)
      W1.append(temp_row)
#####
## (beta'1, beta'2 to solve delta(x_j+d, x_k+d)
for j in range(d+1, 2*d+1):
  for k in range(d+1, 2*d+1):  
    for q in range(2): 
      temp_row = []
      for i in range(1, 3*d+1):
          w = 0
          if j == i and j!=k:
                w = -1/eps
          if k == i and j!=k:
                w = 1/eps      
          temp_row.append(w)
      W1.append(temp_row)
#####
## x_j as identity map 
for j in range(1, 3*d+1):
      temp_row = []
      for k in range(1, 3*d+1):
          w = 0
          if j == k:
                w = 1
          temp_row.append(w)
      W1.append(temp_row)
#####
# for i in W1:
#       print(i)
###########
# Bias matrix for second layer/first hidden layer
############
B1 = []
# bias matrix for eta1, eta2 to solve delta(x_j, x_{j+d})

for j in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=1
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for eta3, eta4 to solve delta(x_j, x_{j+d})

for j in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=1
          temp_row.append(b)
      B1.append(temp_row) 
      
# bias matrix for alpha1, alpha2 to solve delta(x_j, x_k)

for j in range(1, d+1):
  for k in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0 and j!=k:
              b=1
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta1, beta2 to solve delta(x_j, x_k)

for j in range(1, d+1):
  for k in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0 and j!=k:
              b=1
          temp_row.append(b)
      B1.append(temp_row)
############
# bias matrix for alpha'1, alpha'2 to solve delta(x_j+d, x_k+d)

for j in range(d+1, 2*d+1):
  for k in range(d+1, 2*d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0 and j!=k:
              b=1
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta'1, beta'2 to solve delta(x_j+d, x_k+d)

for j in range(d+1, 2*d+1):
  for k in range(d+1, 2*d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0 and j!=k:
              b=1
          temp_row.append(b)
      B1.append(temp_row)
############
# bias matrix for x_j as identity map 

for l in range(1, 3*d+1):
      temp_row = []
      for i in range(1):
          b = 0
          temp_row.append(b)
      B1.append(temp_row)
############
# for i in B1:
#     print(i)
# ##################################
L1 = []  #eta1, eta2, eta3, eta4, alpha1, alpha2, beta1, beta2
for i in range(len(W1)):
    temp_row = []
    L1_i_entry = np.maximum((np.dot(W1[i], X)+B1[i]), 0)
    L1.append(L1_i_entry)
# ############
# print('Printing eta1, eta2, eta3, eta4, alpha1, alpha2, beta1, beta2 nodes of second layer/first hidden layer')
# for i in L1:
#       print(i)
############
# To construct weight matrix for third layer/second hidden layer is L2=W2*L1+B2
W2 = []
# for eta5 that corresponding to delta(xj,x{j+d})
A1 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =1
                else:
                    w=-1
            temp_row.append(w)
        A1.append(temp_row)
#####
A2 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =1
                else:
                    w=-1
            temp_row.append(w)
        A2.append(temp_row)
##########
A3 = []
for j in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A3.append(temp_row)
##########
A4 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A4.append(temp_row)
##########
A5 = []
for j in range(1, d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# alpha'1, alpha'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A5.append(temp_row)
##########
A6 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# beta'1, beta'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A6.append(temp_row)
##########
A7 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, 3*d+1):# xj 
            w = 0
            temp_row.append(w)
        A7.append(temp_row)
##########
for i in range(len(A1)):
    concatenated_row = A1[i] + A2[i]+ A3[i] + A4[i]+A5[i] + A6[i]+A7[i]
    W2.append(concatenated_row)
# print("weight matrix for third layer/second hidden layer")
# print(W2)
#######################
# for psi1 nodes that corresponding to ((1-\delta(x_j, x_{j+d}) ) \land  \delta(x_j, x_{k})\land \delta(x_k, x_{k+d}) , 0 \)
A8 = []
for j in range(1, d+1):
    for k in range(1, d+1):
        temp_row = []
        for l in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if l == j:
                if q==0:
                  w =-1
                else:
                    w=1
            temp_row.append(w)
        A8.append(temp_row)
#####
A9 = []
for j in range(1, d+1):
    for k in range(1, d+1):
        temp_row = []
        for l in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if l == j:
                if q==0:
                  w =-1
                else:
                    w=1
            temp_row.append(w)
        A9.append(temp_row)
##########
A10 = []
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j==k:
                if q==0:
                    w=1
                else:
                    w=-1
            temp_row.append(w)
        A10.append(temp_row)
##########
A11 = []
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j==k:
                if q==0:
                    w=1
                else:
                    w=-1
            temp_row.append(w)
        A11.append(temp_row)
##########
A12 = []
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# alpha'1, alpha'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            if l+d==p and j+d==k:
                if q==0:
                    w=1
                else:
                    w=-1
            temp_row.append(w)
        A12.append(temp_row)
##########
A13 = []
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# beta'1, beta'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            if l+d==p and j+d==k:
                if q==0:
                    w=1
                else:
                    w=-1
            temp_row.append(w)
        A13.append(temp_row)
##########
A14 = []
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, 3*d+1):# xj
             w = 0
             temp_row.append(w)
        A14.append(temp_row)
##########
for i in range(len(A8)):
    concatenated_row = A8[i]+A9[i]+A10[i]+A11[i]+A12[i]+A13[i]+A14[i]
    W2.append(concatenated_row)
# #######################
# for xj as identity map 
A16 = []
for l in range(1, 3*d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        A16.append(temp_row)
#####
A17 = []
for l in range(1, 3*d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        A17.append(temp_row)
##########
A18 = []
for l in range(1, 3*d+1):
        temp_row = []
        for p in range(1, d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A18.append(temp_row)
##########
A19 = []
for l in range(1, 3*d+1):
        temp_row = []
        for p in range(1, d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A19.append(temp_row)
##########
A20 = []
for l in range(1, 3*d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# alpha'1, alpha'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A20.append(temp_row)
##########
A21 = []
for l in range(1, 3*d+1):
        temp_row = []
        for p in range(d+1, 2*d+1):# beta'1, beta'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A21.append(temp_row)
##########
A22 = []
for l in range(1, 3*d+1):
        temp_row = []
        for p in range(1, 3*d+1):# xj
             w = 0
             if l==p:
                 w=1
             temp_row.append(w)
        A22.append(temp_row)
##########
for i in range(len(A16)):
    concatenated_row = A16[i]+A17[i]+ A18[i]+A19[i]+A20[i]+A21[i]+A22[i]
    W2.append(concatenated_row)
# #######################
# for d' nodes that corresponding to summation \delta(x_k, x_{k+d})
A23 = []
for l in range(1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if q==0:
                  w =1
            else:
                    w=-1
            temp_row.append(w)
        A23.append(temp_row)
#####
A24 = []
for l in range(1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if q==0:
                  w = 1
            else:
                    w= -1
            temp_row.append(w)
        A24.append(temp_row)
##########
A25 = []
for l in range(1):
        temp_row = []
        for p in range(1, d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A25.append(temp_row)
##########
A26 = []
for l in range(1):
        temp_row = []
        for p in range(1, d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A26.append(temp_row)
##########
A27 = []
for l in range(1):
        temp_row = []
        for p in range(d+1, 2*d+1):# alpha'1, alpha'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A27.append(temp_row)
##########
A28 = []
for l in range(1):
        temp_row = []
        for p in range(d+1, 2*d+1):# beta'1, beta'2 nodes
          for k in range(d+1, 2*d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A28.append(temp_row)
##########
A29 = []
for l in range(1):
        temp_row = []
        for p in range(1, 3*d+1):# xj
             w = 0
             temp_row.append(w)
        A29.append(temp_row)
##########
for i in range(len(A23)):
    concatenated_row = A23[i]+A24[i]+ A25[i]+A26[i]+A27[i]+A28[i]+A29[i]
    W2.append(concatenated_row)
# # print("weight matrix for third layer/second hidden layer")
# # print(W2)
# #######################
# Bias matrix for third layer/second hidden layer

B2 = []
# bias matrix for eta5 that corresponding to delta(xj,x{j+d})
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -1
          temp_row.append(b)
        B2.append(temp_row)
# bias matrix for psi1 that corresponding to ((1-\delta(x_j, x_{j+d}) ) \land  \delta(x_j, x_{k})\land \delta(x_k, x_{k+d}) , 0 \)
for l in range(1, d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -2
          temp_row.append(b)
        B2.append(temp_row)       
# bias matrix for xj
for l in range(1, 3*d+1):
        temp_row = []
        for k in range(1):
          b = 0
          temp_row.append(b)
        B2.append(temp_row)     
# bias for d'
for i in range(1):
      temp_row = []
      for k in range(1):
        b = -d
        temp_row.append(b)
      B2.append(temp_row)
#################################
L2 = []  # eta5, psi1, d' nodes
for i in range(len(W2)):
    temp_row = []
    L2_i_entry = np.maximum((np.dot(W2[i], L1)+B2[i]), 0)
    L2.append(L2_i_entry)
# ##################################
# print('Printing eta5, psi1(e_jk), d prime nodes for third layer/second hidden layer')
# for i in L2:
#     print(i)
###################
d1 = L2[-1]  # this gets the last element that is d'
# print(d1)
d2 = int(d1[0])
# print(d2)
###########################
# To construct weight matrix for fourth layer/third hidden layer is L3=W3*L2+B3
W3 = []
# tau1 nodes for e'_j
A21 = []
for k in range(1, d+1):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A21.append(temp_row)
# ##########
A22 = []
for j in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):# psi1 for ((1-\delta(x_j, x_{j+d}) ) \land  \delta(x_j, x_{k})\land \delta(x_k, x_{k+d}) , 0 \)
      for k in range(1, d+1):
            w = 0
            if j == l and k<j:
              w = -C
            temp_row.append(w)
    A22.append(temp_row)
##########
A23 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            if i==j:
                w=1
            temp_row.append(w)
    A23.append(temp_row)
##########
A24 = []
for i in range(1, d+1):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A24.append(temp_row)
##########
for i in range(len(A21)):
    concatenated_row = A21[i] +A22[i] + A23[i] + A24[i]
    W3.append(concatenated_row)
# print(W3)    
# #######################
# tau2 nodes for g1_j( \max(( x_{j+2d} - C(1-\delta(x_j, x_{j+d})), 0) )
A25 = []
for k in range(1, d+1):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            if k==p:
                w=C
            temp_row.append(w)
    A25.append(temp_row)
# ##########
A26 = []
for j in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for k in range(1, d+1):
            w = 0
            temp_row.append(w)
    A26.append(temp_row)
##########
A27 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            if i+2*d==j:
                w=1
            temp_row.append(w)
    A27.append(temp_row)
##########
A28 = []
for i in range(1, d+1):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A28.append(temp_row)
##########
for i in range(len(A25)):
    concatenated_row = A25[i] +A26[i] + A27[i] + A28[i]
    W3.append(concatenated_row)
# print(W3)
##################
# tau3 nodes for g2_j( \max(B - C(\delta(x_j, x_{j+d})), 0) )
A29 = []
for k in range(1, d+1):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            if k==p:
                w=-C
            temp_row.append(w)
    A29.append(temp_row)
# ##########
A30 = []
for j in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):# psi1 for delta(x_j, x_{k})\land \delta(x_j+d, x_{k+d})
      for k in range(1, d+1):
            w = 0
            temp_row.append(w)
    A30.append(temp_row)
##########
A31 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A31.append(temp_row)
##########
A32 = []
for i in range(1, d+1):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A32.append(temp_row)
##########
for i in range(len(A29)):
    concatenated_row = A29[i] +A30[i] + A31[i] + A32[i]
    W3.append(concatenated_row)
# print(W3)
##################
# eta7,eta8 nodes to find H(n+d'-x{j+d})
A33 = []
for k in range(1, d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A33.append(temp_row)
# ##########
A34 = []
for k in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 for delta(x_j, x_{k})\land \delta(x_j+d, x_{k+d})
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A34.append(temp_row)
##########
A35 = []
for k in range(1, d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            if k+d==j:
                w=1/eps
            temp_row.append(w)
    A35.append(temp_row)
##########
A36 = []
for k in range(1, d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A36.append(temp_row)
##########
for i in range(len(A33)):
    concatenated_row = A33[i] +A34[i] + A35[i] + A36[i]
    W3.append(concatenated_row)
##################
# eta9,eta10 nodes to find H(n-i)
A33 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A33.append(temp_row)
# ##########
A34 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A34.append(temp_row)
##########
A35 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A35.append(temp_row)
##########
A36 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A36.append(temp_row)
##########
for i in range(len(A33)):
    concatenated_row = A33[i] +A34[i] + A35[i] + A36[i]
    W3.append(concatenated_row)
##################
# eta11,eta12 nodes to find H(i-n-1)
A37 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A37.append(temp_row)
# ##########
A38 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 for delta(x_j, x_{k})\land \delta(x_j+d, x_{k+d})
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A38.append(temp_row)
##########
A39 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A39.append(temp_row)
##########
A40 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A40.append(temp_row)
##########
for i in range(len(A37)):
    concatenated_row = A37[i] +A38[i] + A39[i] + A40[i]
    W3.append(concatenated_row)
##################
# eta13,eta14 nodes to find H(n+d'-i)
A41 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A41.append(temp_row)
# ##########
A42 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 for delta(x_j, x_{k})\land \delta(x_j+d, x_{k+d})
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A42.append(temp_row)
##########
A43 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A43.append(temp_row)
##########
A44 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A44.append(temp_row)
##########
for i in range(len(A41)):
    concatenated_row = A41[i] +A42[i] + A43[i] + A44[i]
    W3.append(concatenated_row)
##################
# eta15,eta16 nodes to find H(n+d'-k)
A45 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A45.append(temp_row)
# ##########
A46 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A46.append(temp_row)
##########
A47 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A47.append(temp_row)
##########
A48 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A48.append(temp_row)
##########
for i in range(len(A45)):
    concatenated_row = A45[i] +A46[i] + A47[i] + A48[i]
    W3.append(concatenated_row)
##################
# eta17,eta18 nodes to find H(k+n-1)nn
A49 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A49.append(temp_row)
# ##########
A50 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A50.append(temp_row)
##########
A51 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A51.append(temp_row)
##########
A52 = []
for k in range(1, n+d+1):
  for q in range(2):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A52.append(temp_row)
##########
for i in range(len(A49)):
    concatenated_row = A49[i] +A50[i] + A51[i] + A52[i]
    W3.append(concatenated_row)
##################
# eta5 nodes as identity map
A53 = []
for k in range(1, d+1):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            if k==p:
                w=1
            temp_row.append(w)
    A53.append(temp_row)
# ##########
A54 = []
for k in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A54.append(temp_row)
##########
A55 = []
for k in range(1, d+1):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            temp_row.append(w)
    A55.append(temp_row)
##########
A56 = []
for k in range(1, d+1):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A56.append(temp_row)
##########
for i in range(len(A53)):
    concatenated_row = A53[i] +A54[i] + A55[i] + A56[i]
    W3.append(concatenated_row)
##################
# x_{j+d}
A57 = []
for k in range(d+1, 2*d+1):
    temp_row = []
    for p in range(1, d+1):# eta5 that corresponding to delta(xj,x{j+d})
            w = 0
            temp_row.append(w)
    A57.append(temp_row)
# ##########
A58 = []
for k in range(d+1, 2*d+1):
    temp_row = []
    for l in range(1, d+1):# psi1 
      for i in range(1, d+1):
            w = 0
            temp_row.append(w)
    A58.append(temp_row)
##########
A59 = []
for k in range(d+1, 2*d+1):
    temp_row = []
    for j in range(1, 3*d+1):# xj
            w = 0
            if j==k:
                w=1
            temp_row.append(w)
    A59.append(temp_row)
##########
A60 = []
for k in range(d+1, 2*d+1):
    temp_row = []
    for p in range(1):
            w = 0
            temp_row.append(w)
    A60.append(temp_row)
##########
for i in range(len(A57)):
    concatenated_row = A57[i] +A58[i] + A59[i] + A60[i]
    W3.append(concatenated_row)
##################
# Bias matrix for fourth layer/third hidden layer

B3 = [] 
# bias matrix for tau1
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = 0
          temp_row.append(b)
        B3.append(temp_row)      
# bias matrix for tau2
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -C
          temp_row.append(b)
        B3.append(temp_row) 
# bias matrix for tau3
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = B
          temp_row.append(b)
        B3.append(temp_row)
# bias matrix for eta7,eta8
for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(-n-d2-1)/eps
          else:
              b=(-n-d2-1)/eps
          temp_row.append(b)
        B3.append(temp_row) 
# bias matrix for eta9,eta10
for i in range(1, n+d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(n-i)/eps
          else:
              b=(n-i)/eps
          temp_row.append(b)
        B3.append(temp_row)         
################## 
# bias matrix for eta11,eta12
for i in range(1, n+d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(i-n-1)/eps
          else:
              b=(i-n-1)/eps
          temp_row.append(b)
        B3.append(temp_row)         
################## 
# bias matrix for eta13,eta14
for i in range(1, n+d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(n+d2-i)/eps
          else:
              b=(n+d2-i)/eps
          temp_row.append(b)
        B3.append(temp_row)         
################## 
# bias matrix for eta15,eta16
for k in range(1, n+d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
          b = 0
          if q==0:
             b=1+(n+d2-k)/eps
          else:
              b=(n+d2-k)/eps
          temp_row.append(b)
        B3.append(temp_row)         
################## 
# bias matrix for eta17,eta18
for k in range(1, n+d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
          b = 0
          if q==0:
             b=1+(k-n-1)/eps
          else:
              b=(k-n-1)/eps
          temp_row.append(b)
        B3.append(temp_row)     
##################
# bias matrix for eta5 as identity map
for k in range(1, d+1):
        temp_row = []
        for i in range(1):
          b = 0
          temp_row.append(b)
        B3.append(temp_row)
##################
# bias matrix for x_j+d
for k in range(1, d+1):
        temp_row = []
        for i in range(1):
          b = 0
          temp_row.append(b)
        B3.append(temp_row)
######################        
# print('Printing B3')
# for i in B3:
#     print(i)     
##################################
L3 = []  # tau1,tau2,tau3,eta7,eta8,eta9,eta10,eta11,eta12,eta13,eta14,eta15,eta16,eta17,eta18 nodes
for i in range(len(W3)):
    temp_row = []
    L3_i_entry = np.maximum((np.dot(W3[i], L2)+B3[i]), 0)
    L3.append(L3_i_entry)
##################################
# print('Printing tau1,tau2,tau3,eta7,eta8,eta9,eta10,eta11,eta12,eta13,eta14,eta15,eta16,eta17,eta18 nodes for fourth layer/third hidden layer')
# for i in L3:
#     print(i)
###########################
# To construct weight matrix for fifth layer/fouth hidden layer is L4=W4*L3+B4
W4 = []
# tau1(e') nodes as identity map
D1 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                if i == j:
                    w = 1
                temp_row.append(w)
            D1.append(temp_row)

##########
D2 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D2.append(temp_row)
##########
D3 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D3.append(temp_row)
##########
D4 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D4.append(temp_row)
##########
D5 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D5.append(temp_row)
##########
D6 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D6.append(temp_row)
##########
D7 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D7.append(temp_row)
##########
D8 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D8.append(temp_row)
##########
D9 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D9.append(temp_row)
##########
D10 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
                        w = 0
                        temp_row.append(w)
            D10.append(temp_row)
##########
D11 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D11.append(temp_row)
##########
for i in range(len(D1)):
    concatenated_row = D1[i]+D2[i]+D3[i]+D4[i]+D5[i]+D6[i]+D7[i]+D8[i]+D9[i]+D10[i]+D11[i] 
    W4.append(concatenated_row)
#######################
# alpha3, alpha4 nodes to calculate delta(e'_j, x_{j+d}) 
D12 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                if i == j:
                    w = 1/eps
                temp_row.append(w)
            D12.append(temp_row)

##########
D13 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D13.append(temp_row)
##########
D14 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D14.append(temp_row)
##########
D15 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D15.append(temp_row)
##########
D16 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D16.append(temp_row)
##########
D17 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D17.append(temp_row)
##########
D18 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D18.append(temp_row)
##########
D19 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D19.append(temp_row)
##########
D20 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D20.append(temp_row)
##########
D21 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
                        w = 0
                        temp_row.append(w)
            D21.append(temp_row)
##########
D22 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                if i+d == j:
                    w = -1/eps
                temp_row.append(w)
            D22.append(temp_row)
##########
for i in range(len(D12)):
    concatenated_row = D12[i]+D13[i]+D14[i]+D15[i]+D16[i]+D17[i]+D18[i]+D19[i]+D20[i]+D21[i]+D22[i]  
    W4.append(concatenated_row)
#######################
# beta3, beta4 nodes to calculate delta(e'_j, x_{j+d}) 
D23 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                if i == j:
                    w = -1/eps
                temp_row.append(w)
            D23.append(temp_row)

##########
D24 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D24.append(temp_row)
##########
D25 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D25.append(temp_row)
##########
D26 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D26.append(temp_row)
##########
D27 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D27.append(temp_row)
##########
D28 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D28.append(temp_row)
##########
D29 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D29.append(temp_row)
##########
D30 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D30.append(temp_row)
##########
D31 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D31.append(temp_row)
##########
D32 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
                        w = 0
                        temp_row.append(w)
            D32.append(temp_row)
##########
D33 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                if i+d == j:
                    w = 1/eps
                temp_row.append(w)
            D33.append(temp_row)
##########
for i in range(len(D23)):
    concatenated_row = D23[i]+D24[i]+D25[i]+D26[i]+D27[i]+D28[i]+D29[i]+D30[i]+D31[i]+D32[i]+D33[i]
    W4.append(concatenated_row)
#######################
# eta19, eta20 nodes to calculate H(-n-d'-1+e'_j) 
D34 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                if i == j:
                    w = 1/eps
                temp_row.append(w)
            D34.append(temp_row)

##########
D35 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D35.append(temp_row)
##########
D36 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D36.append(temp_row)
##########
D37 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D37.append(temp_row)
##########
D38 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D38.append(temp_row)
##########
D39 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D39.append(temp_row)
##########
D40 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D40.append(temp_row)
##########
D41 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D41.append(temp_row)
##########
D42 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for p in range(2):
                w = 0
                temp_row.append(w)
            D42.append(temp_row)
##########
D43 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
                        w = 0
                        temp_row.append(w)
            D43.append(temp_row)
##########
D44 = []
for i in range(1, d+1):
    for q in range(2):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D44.append(temp_row)
##########
for i in range(len(D34)):
    concatenated_row = D34[i]+D35[i]+D36[i]+D37[i]+D38[i]+D39[i]+D40[i]+D41[i]+D42[i]+D43[i]+D44[i]
    W4.append(concatenated_row)
#######################
# eta21 nodes to calculate (1-\delta(x_j, x_{j+d})) \land H(n+d'-x_{j+d}) 
D45 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D45.append(temp_row)

##########
D46 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D46.append(temp_row)
##########
D47 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D47.append(temp_row)
##########
D48 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        if i == j:
                          if p==0:
                            w=1
                          else:
                              w=-1
                        temp_row.append(w)
            D48.append(temp_row)
##########
D49 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D49.append(temp_row)
##########
D50 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D50.append(temp_row)
##########
D51 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D51.append(temp_row)
##########
D52 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D52.append(temp_row)
##########
D53 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D53.append(temp_row)
##########
D54 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
                        w = 0
                        if i == j:
                            w = -1
                        temp_row.append(w)
            D54.append(temp_row)
##########
D55 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D55.append(temp_row)
##########
for i in range(len(D45)):
    concatenated_row = D45[i]+D46[i]+D47[i]+D48[i]+D49[i]+D50[i]+D51[i]+D52[i]+D53[i]+D54[i]+D55[i]
    W4.append(concatenated_row)
#######################
# tau4 nodes correspond to gj
D56 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D56.append(temp_row)

##########
D57 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                if i == j:
                    w=1
                temp_row.append(w)
            D57.append(temp_row)
##########
D58 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        if i == j:
                            w=1
                        temp_row.append(w)
            D58.append(temp_row)
##########
D59 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D59.append(temp_row)
##########
D60 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D60.append(temp_row)
##########
D61 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D61.append(temp_row)
##########
D62 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D62.append(temp_row)
##########
D63 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D63.append(temp_row)
##########
D64 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D64.append(temp_row)
##########
D65 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D65.append(temp_row)
##########
D66 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D66.append(temp_row)
##########
for i in range(len(D56)):
    concatenated_row = D56[i]+D57[i]+D58[i]+D59[i]+D60[i]+D61[i]+D62[i]+D63[i]+D64[i]+D65[i]+D66[i]
    W4.append(concatenated_row)
#######################
# eta22 nodes for H(n-i)
D67 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D67.append(temp_row)

##########
D68 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D68.append(temp_row)
##########
D69 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D69.append(temp_row)
##########
D70 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D70.append(temp_row)
##########
D71 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                if i == j:
                  if q==0:  
                    w=1
                  else:
                      w=-1
                temp_row.append(w)
            D71.append(temp_row)
##########
D72 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D72.append(temp_row)
##########
D73 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D73.append(temp_row)
##########
D74 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D74.append(temp_row)
##########
D75 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D75.append(temp_row)
##########
D76 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D76.append(temp_row)
##########
D77 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D77.append(temp_row)
##########
for i in range(len(D67)):
    concatenated_row = D67[i]+D68[i]+D69[i]+D70[i]+D71[i]+D72[i]+D73[i]+D74[i]+D75[i]+D76[i]+D77[i]
    W4.append(concatenated_row)
#######################
# eta23 nodes for H(i-n-1)
D78 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D78.append(temp_row)

##########
D79 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D79.append(temp_row)
##########
D80 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D80.append(temp_row)
##########
D81 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D81.append(temp_row)
##########
D82 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D82.append(temp_row)
##########
D83 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        if i == j:
                          if q==0:  
                            w=1
                          else:
                              w=-1
                        temp_row.append(w)
            D83.append(temp_row)
##########
D84 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D84.append(temp_row)
##########
D85 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D85.append(temp_row)
##########
D86 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D86.append(temp_row)
##########
D87 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D87.append(temp_row)
##########
D88 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D88.append(temp_row)
##########
for i in range(len(D78)):
    concatenated_row = D78[i]+D79[i]+D80[i]+D81[i]+D82[i]+D83[i]+D84[i]+D85[i]+D86[i]+D87[i]+D88[i]
    W4.append(concatenated_row)
#######################
# phi1 nodes for ( H(n+d' - i) \land H(i-n-1) \land H(n+d' - k))
D89 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D89.append(temp_row)

##########
D90 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D90.append(temp_row)
##########
D91 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D91.append(temp_row)
##########
D92 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D92.append(temp_row)
##########
D93 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D93.append(temp_row)
##########
D94 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        if i == j:
                          if q==0:  
                            w=1
                          else:
                              w=-1
                        temp_row.append(w)
            D94.append(temp_row)
##########
D95 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                if i == j:
                  if q==0:  
                    w=1
                  else:
                      w=-1
                temp_row.append(w)
            D95.append(temp_row)
##########
D96 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        if k == j:
                          if q==0:  
                            w=1
                          else:
                              w=-1
                        temp_row.append(w)
            D96.append(temp_row)
##########
D97 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D97.append(temp_row)
##########
D98 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D98.append(temp_row)
##########
D99 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D99.append(temp_row)
##########
for i in range(len(D89)):
    concatenated_row = D89[i]+D90[i]+D91[i]+D92[i]+D93[i]+D94[i]+D95[i]+D96[i]+D97[i]+D98[i]+D99[i]
    W4.append(concatenated_row)
######################
# phi2 nodes for ( H(n+d' - k) \land H(k-n-1) \land H(n-i))
D100 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D100.append(temp_row)
##########
D101 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D101.append(temp_row)
##########
D102 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D102.append(temp_row)
##########
D103 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D103.append(temp_row)
##########
D104 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                if i == j:
                  if q==0:  
                    w=1
                  else:
                      w=-1
                temp_row.append(w)
            D104.append(temp_row)
##########
D105= []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D105.append(temp_row)
##########
D106 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D106.append(temp_row)
##########
D107 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        if k == j:
                          if q==0:  
                            w=1
                          else:
                              w=-1
                        temp_row.append(w)
            D107.append(temp_row)
##########
D108 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                if k == j:
                  if q==0:  
                    w=1
                  else:
                      w=-1
                temp_row.append(w)
            D108.append(temp_row)
##########
D109 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D109.append(temp_row)
##########
D110 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                temp_row.append(w)
            D110.append(temp_row)
##########
for i in range(len(D100)):
    concatenated_row = D100[i]+D101[i]+D102[i]+D103[i]+D104[i]+D105[i]+D106[i]+D107[i]+D108[i]+D109[i]+D110[i]
    W4.append(concatenated_row)
#######################
# x_j+d nodes 
D111 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, d+1):#tau1
                w = 0
                temp_row.append(w)
            D111 .append(temp_row)

##########
D112  = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, d+1):# tau2
                w = 0
                temp_row.append(w)
            D112.append(temp_row)
##########
D113 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, d+1):  # tau3 nodes
                        w = 0
                        temp_row.append(w)
            D113.append(temp_row)
##########
D114 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, d+1):  # eta7,eta8 nodes
                for p in range(2):
                        w = 0
                        temp_row.append(w)
            D114.append(temp_row)
##########
D115 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta9,eta10 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D115.append(temp_row)
##########
D116= []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta11,eta12 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D116.append(temp_row)
##########
D117 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta13,eta14 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D117.append(temp_row)
##########
D118 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta15,eta16 nodes
                for q in range(2):
                        w = 0
                        temp_row.append(w)
            D118.append(temp_row)
##########
D119 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):  # eta17,eta18 nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D119.append(temp_row)
##########
D120 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(1, d+1):  # eta5 nodes
              w = 0
              temp_row.append(w)
            D120.append(temp_row)
##########
D121 = []
for i in range(d+1, 2*d+1):
            temp_row = []
            for j in range(d+1, 2*d+1):  # x_j+d nodes
                w = 0
                if i==j:
                    w=1
                temp_row.append(w)
            D121.append(temp_row)
##########
for i in range(len(D111)):
    concatenated_row = D111[i]+D112[i]+D113[i]+D114[i]+D115[i]+D116[i]+D117[i]+D118[i]+D119[i]+D120[i]+D121[i]
    W4.append(concatenated_row)
#########################
# print("weight matrix for fifth layer/fourth hidden layer")
# print(W4)
#####################
# Bias matrix for fifth layer/fourth hidden layer

B4 = []

# bias matrix for tau1 nodes 

for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = 0
          temp_row.append(b)
        B4.append(temp_row)

# # bias matrix for alpha3, alpha4
for i in range(1, d+1):
  for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b=1
                temp_row.append(b)
            B4.append(temp_row) 
            
# # bias matrix for beta3, beta4
for i in range(1, d+1):
  for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b=1
                temp_row.append(b)
            B4.append(temp_row)
# # bias matrix for eta19, eta20
for i in range(1, d+1):
  for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b=1+(-n-d2-1)/eps
                else:
                    b=(-n-d2-1)/eps
                temp_row.append(b)
            B4.append(temp_row) 
# # bias matrix for eta21
for i in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row) 
# # bias matrix for tau4
for i in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row) 
# # bias matrix for eta22
for i in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row)
# # bias matrix for eta23
for i in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row)  
# # bias matrix for psi1
for i in range(1, n+d+1):
    for k in range(1, n+d+1): 
         temp_row = []
         for k in range(1):
             b = -2
             temp_row.append(b)
         B4.append(temp_row) 
# # bias matrix for psi2
for i in range(1, n+d+1):
    for k in range(1, n+d+1): 
          temp_row = []
          for k in range(1):
              b = -2
              temp_row.append(b)
          B4.append(temp_row) 
# # bias matrix for x_j+d
for i in range(d+1, 2*d+1): 
          temp_row = []
          for k in range(1):
              b = 0
              temp_row.append(b)
          B4.append(temp_row)          
###################################
# print('Printing B4')
# for i in B4:
#     print(i)
###################################
L4 = []  # alpha3, alpha4, beta3, beta4, eta19, eta20, eta21, tau4, eta22, eta23, phi1, phi2 nodes
for i in range(len(W4)):
    temp_row = []
    L4_i_entry = np.maximum((np.dot(W4[i], L3)+B4[i]), 0)
    L4.append(L4_i_entry)
###################################
# print('Printing alpha3, alpha4, beta3, beta4, eta19, eta20, eta21, tau4, eta22, eta23, phi1, phi2 nodes for fifth layer/fouth hidden hidden layer')
# for i in L4:
#     print(i)
##################
# To construct weight matrix for sixth layer/fifth hidden layer is L5=W5*L4+B5
W5 = []
# tau1 nodes as identity map
E1 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# tau1 nodes
                w = 0
                if i==k:
                    w=1
                temp_row.append(w)
        E1.append(temp_row)
##########
E2 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# alpha3, alpha4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E2.append(temp_row)

##########
E3 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# beta3, beta4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E3.append(temp_row)
##########
E4 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta19, eta20
              for q in range(2):
                w = 0
                temp_row.append(w)
        E4.append(temp_row)
##########
E5 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E5.append(temp_row)
##########
E6 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E6.append(temp_row)
##########
E7 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E7.append(temp_row)
##########
E8 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E8.append(temp_row)

##########
E9 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi1
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E9.append(temp_row)
##########
E10 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi2
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E10.append(temp_row)
##########
E11 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E11.append(temp_row)
##########
for i in range(len(E1)):
    concatenated_row = E1[i] + E2[i] + E3[i] + E4[i] + E5[i] + E6[i] + E7[i] + E8[i] + E9[i] + E10[i] + E11[i] 
    W5.append(concatenated_row)
#######################
# tau6 nodes for (1-\delta(e'_j, x_{j+d}) )\land H(n+d'-e'_j)
E12 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E12.append(temp_row)
##########
E13 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# alpha3, alpha4 nodes
          for q in range(2):
                w = 0
                if i==k:
                  if q==0:
                      w=-1
                  else:
                      w=1
                temp_row.append(w)
        E13.append(temp_row)

##########
E14 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# beta3, beta4 nodes
          for q in range(2):
                w = 0
                if i==k:
                  if q==0:
                      w=-1
                  else:
                      w=1
                temp_row.append(w)
        E14.append(temp_row)
##########
E15 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta19, eta20
              for q in range(2):
                w = 0
                if i==k:
                  if q==0:
                      w=1
                  else:
                      w=-1
                temp_row.append(w)
        E15.append(temp_row)
##########
E16 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E16.append(temp_row)
##########
E17 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E17.append(temp_row)
##########
E18 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E18.append(temp_row)
##########
E19 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E19.append(temp_row)
##########
E20 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#psi1
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E20.append(temp_row)
##########
E21 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#psi2
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E21.append(temp_row)
##########
E22 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E22.append(temp_row)
##########
for i in range(len(E12)):
    concatenated_row = E12[i] + E13[i] + E14[i] + E15[i] + E16[i] + E17[i] + E18[i] + E19[i] + E20[i] +E21[i] +E22[i] 
    W5.append(concatenated_row)
######################
# tau4 nodes as identity map
E23 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E23.append(temp_row)
##########
E24 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# alpha3, alpha4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E24.append(temp_row)

##########
E25 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# beta3, beta4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E25.append(temp_row)
##########
E26 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta19, eta20
              for q in range(2):
                w = 0
                temp_row.append(w)
        E26.append(temp_row)
##########
E27 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E27.append(temp_row)
##########
E28 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#tau4
                w = 0
                if i==k:
                      w=1
                temp_row.append(w)
        E28.append(temp_row)
##########
E29 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E29.append(temp_row)
##########
E30 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E30.append(temp_row)
##########
E31 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi1
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E31.append(temp_row)
##########
E32 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi2
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E32.append(temp_row)
##########
E33 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E33.append(temp_row)
##########
for i in range(len(E23)):
    concatenated_row = E23[i]+E24[i]+E25[i]+E26[i]+E27[i]+E28[i]+E29[i]+E30[i]+E31[i]+ E32[i]+E33[i]  
    W5.append(concatenated_row)
#######################
# eta22 nodes as identity map
E34 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E34.append(temp_row)
##########
E35 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# alpha3, alpha4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E35.append(temp_row)

##########
E36 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# beta3, beta4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E36.append(temp_row)
##########
E37 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#eta19, eta20
              for q in range(2):
                w = 0
                temp_row.append(w)
        E37.append(temp_row)
##########
E38 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E38.append(temp_row)
##########
E39 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E39.append(temp_row)
##########
E40 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta22
                w = 0
                if i==k:
                      w=1
                temp_row.append(w)
        E40.append(temp_row)
##########
E41 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E41.append(temp_row)
##########
E42 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi1
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E42.append(temp_row)
##########
E43 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi2
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E43.append(temp_row)
##########
E44 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E44.append(temp_row)
##########
for i in range(len(E34)):
    concatenated_row = E34[i] + E35[i] + E36[i] + E37[i] + E38[i] + E39[i] + E40[i]+E41[i] + E42[i] + E43[i] +  E44[i]
    W5.append(concatenated_row)
#######################
# eta23 nodes as identity map
E45 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E45.append(temp_row)
##########
E46 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# alpha3, alpha4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E46.append(temp_row)
##########
E47 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):# beta3, beta4 nodes
          for q in range(2):
                w = 0
                temp_row.append(w)
        E47.append(temp_row)
##########
E48 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#eta19, eta20
              for q in range(2):
                w = 0
                temp_row.append(w)
        E48.append(temp_row)
##########
E49 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E49.append(temp_row)
##########
E50 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E50.append(temp_row)
##########
E51 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E51.append(temp_row)
##########
E52 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#eta23
                w = 0
                if i==k:
                      w=1
                temp_row.append(w)
        E52.append(temp_row)
##########
E53 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi1
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E53.append(temp_row)
##########
E54 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#phi2
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E54.append(temp_row)
##########
E55 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E55.append(temp_row)
##########
for i in range(len(E45)):
    concatenated_row = E45[i] + E46[i] + E47[i] + E48[i] + E49[i] + E50[i] + E51[i]+E52[i] + E53[i] + E54[i] +  E55[i]
    W5.append(concatenated_row)
#######################
# alpha5,alpha6 nodes
E56 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):  
        temp_row = []
        for i in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E56.append(temp_row)
##########
E57 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):# alpha3, alpha4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E57.append(temp_row)
##########
E58 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):# beta3, beta4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E58.append(temp_row)
##########
E59 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#eta19, eta20
              for p in range(2):
                w = 0
                temp_row.append(w)
        E59.append(temp_row)
##########
E60 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E60.append(temp_row)
##########
E61 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#tau4
                w = 0
                if j==i and j!=k:
                      w=1/eps
                if k==i and j!=k:
                    w=-1/eps
                temp_row.append(w)
        E61.append(temp_row)
##########
E62 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E62.append(temp_row)
##########
E63 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E63.append(temp_row)
##########
E64 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#phi1
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E64.append(temp_row)
##########
E65 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#phi2
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E65.append(temp_row)
##########
E66 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E66.append(temp_row)
##########
for i in range(len(E56)):
    concatenated_row = E56[i] + E57[i] + E58[i] + E59[i] + E60[i] + E61[i] + E62[i]+E63[i] + E64[i] + E65[i] +  E66[i]
    W5.append(concatenated_row)
#######################
# beta5,beta6 nodes
E67 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):  
        temp_row = []
        for i in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E67.append(temp_row)
##########
E68 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):# alpha3, alpha4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E68.append(temp_row)
##########
E69 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):# beta3, beta4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E69.append(temp_row)
##########
E70 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#eta19, eta20
              for p in range(2):
                w = 0
                temp_row.append(w)
        E70.append(temp_row)
##########
E71 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E71.append(temp_row)
##########
E72 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):#tau4
                w = 0
                if j==i and j!=k:
                      w=-1/eps
                if k==i and j!=k:
                    w=1/eps
                temp_row.append(w)
        E72.append(temp_row)
##########
E73 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E73.append(temp_row)
##########
E74 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E74.append(temp_row)
##########
E75 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#phi1
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E75.append(temp_row)
##########
E76 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):#phi2
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E76.append(temp_row)
##########
E77 = []
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E77.append(temp_row)
##########
for i in range(len(E67)):
    concatenated_row = E67[i] + E68[i] + E69[i] + E70[i] + E71[i] + E72[i] + E73[i]+E74[i] + E75[i] + E76[i] +  E77[i]
    W5.append(concatenated_row)
#####################
# phi3 nodes for p'ik
E78 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E78.append(temp_row)
##########
E79 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# alpha3, alpha4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E79.append(temp_row)
##########
E80 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# beta3, beta4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E80.append(temp_row)
##########
E81 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#eta19, eta20
              for p in range(2):
                w = 0
                temp_row.append(w)
        E81.append(temp_row)
##########
E82 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E82.append(temp_row)
##########
E83 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E83.append(temp_row)
##########
E84 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E84.append(temp_row)
##########
E85 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E85.append(temp_row)
##########
E86 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#phi1
          for l in range(1, n+d+1):
                w = 0
                if j==i and k==l:
                    w=C
                temp_row.append(w)
        E86.append(temp_row)
##########
E87 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#phi2
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E87.append(temp_row)
##########
E88 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E88.append(temp_row)
##########
for i in range(len(E78)):
    concatenated_row =E78[i]+E79[i]+E80[i] + E81[i] + E82[i] + E83[i] + E84[i]+E85[i]+E86[i]+E87[i]+E88[i]
    W5.append(concatenated_row)  
######################## 
# phi4 nodes for q'ik
E89 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E89.append(temp_row)
##########
E90 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# alpha3, alpha4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E90.append(temp_row)
##########
E91 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):# beta3, beta4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E91.append(temp_row)
##########
E92 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#eta19, eta20
              for p in range(2):
                w = 0
                temp_row.append(w)
        E92.append(temp_row)
##########
E93 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#eta21
                w = 0
                temp_row.append(w)
        E93.append(temp_row)
##########
E94 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E94.append(temp_row)
##########
E95 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E95.append(temp_row)
##########
E96 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E96.append(temp_row)
##########
E97 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#psi1
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E97.append(temp_row)
##########
E98 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(1, n+d+1):#psi2
          for l in range(1, n+d+1):
                w = 0
                if j==i and k==l:
                    w=C
                temp_row.append(w)
        E98.append(temp_row)
##########
E99 = []
for j in range(1, n+d+1):
   for k in range(1, n+d+1):
        temp_row = []
        for i in range(d+1, 2*d+1): #x_j+d
                w = 0
                temp_row.append(w)
        E99.append(temp_row)
##########
for i in range(len(E89)):
    concatenated_row =E89[i]+E90[i]+E91[i] + E92[i] + E93[i] + E94[i] + E95[i]+E96[i]+E97[i]+E98[i]+E99[i]
    W5.append(concatenated_row)  
######################## 
# x2 nodes
E100 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):# tau1 nodes
                w = 0
                temp_row.append(w)
        E100.append(temp_row)
##########
E101 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):# alpha3, alpha4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E101.append(temp_row)
##########
E102 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):# beta3, beta4 nodes
          for p in range(2):
                w = 0
                temp_row.append(w)
        E102.append(temp_row)
##########
E103 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):#eta19, eta20
              for p in range(2):
                w = 0
                temp_row.append(w)
        E103.append(temp_row)
##########
E104 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):#eta21
                w = 0
                if j==i:
                    w=-C
                temp_row.append(w)
        E104.append(temp_row)
##########
E105 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, d+1):#tau4
                w = 0
                temp_row.append(w)
        E105.append(temp_row)
##########
E106 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta22
                w = 0
                temp_row.append(w)
        E106.append(temp_row)
##########
E107 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, n+d+1):#eta23
                w = 0
                temp_row.append(w)
        E107.append(temp_row)
##########
E108 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, n+d+1):#phi1
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E108.append(temp_row)
##########
E109 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(1, n+d+1):#phi2
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E109.append(temp_row)
##########
E110 = []
for j in range(1, d+1):
        temp_row = []
        for i in range(d+1, 2*d+1): #x_j+d
                w = 0
                if j+d==i:
                    w=1
                temp_row.append(w)
        E110.append(temp_row)
##########
for i in range(len(E100)):
    concatenated_row =E100[i]+E101[i]+E102[i] + E103[i]+E104[i]+E105[i] + E106[i]+E107[i]+E108[i]+E109[i]+E110[i]
    W5.append(concatenated_row)  
########################    
# print("weight matrix for sixth layer/fifth hidden layer")
# print(W5)
# #####################
# #Bias matrix for sixth layer/fifth hidden layer

B5 = []

# bias matrix for tau1 nodes
for i in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B5.append(temp_row)

# bias matrix for tau6
for i in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = 1
        temp_row.append(b)
      B5.append(temp_row)  

# bias matrix for tau4
for i in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B5.append(temp_row)   

# bias matrix for eta22
for i in range(1, n+d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B5.append(temp_row)
# bias matrix for eta23
for i in range(1, n+d+1):
        temp_row = []
        for j in range(1):
            b = 0
            temp_row.append(b)
        B5.append(temp_row)
# bias for alpha5,alpha6 nodes
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):   
         temp_row = []
         for l in range(1):
             b = 0
             if q==0:
                 b=1
             temp_row.append(b)
         B5.append(temp_row)
# bias for beta5,beta6 nodes
for j in range(1, d+1):
   for k in range(1, d+1):
     for q in range(2):   
         temp_row = []
         for l in range(1):
             b = 0
             if q==0:
                 b=1
             temp_row.append(b)
         B5.append(temp_row) 
# bias for phi3 nodes for p'ik nodes
for j in range(1, n+d+1):
   for k in range(1, n+d+1):   
         temp_row = []
         for l in range(1):
             b = B-C
             temp_row.append(b)
         B5.append(temp_row) 
# bias for phi4 nodes for q'ik nodes
for j in range(1, n+d+1):
   for k in range(1, n+d+1):   
         temp_row = []
         for l in range(1):
             b = B-C
             temp_row.append(b)
         B5.append(temp_row) 
# bias for x2 nodes
for j in range(1, d+1):  
         temp_row = []
         for l in range(1):
             b = 0
             temp_row.append(b)
         B5.append(temp_row)          
##################################
L5 = []  # x2, eta25, alpha5, alpha6, beta5, beta6, psi3, psi4 nodes
for i in range(len(W5)):
    temp_row = []
    L5_i_entry = np.maximum((np.dot(W5[i], L4)+B5[i]), 0)
    L5.append(L5_i_entry)
# ##################################
# print('Printing x2, eta25, alpha5, alpha6, beta5, beta6, psi3, psi4 nodes for sixth layer/fifth hidden layer')
# for i in L5:
#     print(i)
#####################
# To construct weight matrix for seventh layer/sixth hidden layer is L6=W6*L5+B6
W6 = []
# tau4(gj) nodes as identity map
F1 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F1.append(temp_row)
##########
F2 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F2.append(temp_row)

##########
F3 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  if i==j:
                      w=1
                  temp_row.append(w)
            F3.append(temp_row)

##########
F4 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F4.append(temp_row)
##########
F5 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F5.append(temp_row)
##########
F6 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for k in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F6.append(temp_row)
##########
F7 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for k in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F7.append(temp_row)
##########
F8 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for k in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F8.append(temp_row)
##########
F9 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for k in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F9.append(temp_row)
##########
F10 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F10.append(temp_row)
##########
for i in range(len(F1)):
    concatenated_row = F1[i] + F2[i] + F3[i] + F4[i]+F5[i] + F6[i] + F7[i] + F8[i] +F9[i] + F10[i]
    W6.append(concatenated_row)
######################
# tau5(g'j) nodes 
F11= []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F11.append(temp_row)
##########
F12 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F12.append(temp_row)

##########
F13 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F13.append(temp_row)

##########
F14 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F14.append(temp_row)
##########
F15 = []
for i in range(1, d+1):
            temp_row = []
            for k in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F15.append(temp_row)
##########
F16 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for k in range(1, d+1):
                 for q in range(2):
                  w = 0
                  if i==j and k<=(i-1):
                    if q==0:
                        w=1
                    else:
                        w=-1
                  temp_row.append(w)
            F16.append(temp_row)
##########
F17 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for k in range(1, d+1):
                 for q in range(2):
                  w = 0
                  if i==j and k >= i:
                    if q==0:
                        w=-1
                    else:
                        w=1
                  temp_row.append(w)
            F17.append(temp_row)
##########
F18 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for k in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F18.append(temp_row)
##########
F19 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for k in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F19.append(temp_row)
##########
F20 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F20.append(temp_row)
##########
for i in range(len(F11)):
    concatenated_row = F11[i] + F12[i] + F13[i] + F14[i]+F15[i] + F16[i] + F17[i] + F18[i] +F19[i] + F20[i]
    W6.append(concatenated_row)
######################
# mu1(r_ik) nodes 
F21= []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F21.append(temp_row)
##########
F22 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F22.append(temp_row)
##########
F23 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F23.append(temp_row)

##########
F24 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F24.append(temp_row)
##########
F25 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F25.append(temp_row)
##########
F26 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F26.append(temp_row)
##########
F27 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F27.append(temp_row)
##########
F28 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  if i==j and k==l:
                      w=-1
                  temp_row.append(w)
            F28.append(temp_row)
##########
F29 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  if i==j and k==l:
                      w=-1
                  temp_row.append(w)
            F29.append(temp_row)
##########
F30 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F30.append(temp_row)
##########
for i in range(len(F21)):
    concatenated_row = F21[i] + F22[i] + F23[i] + F24[i]+F25[i] + F26[i] + F27[i] + F28[i] +F29[i] + F30[i]
    W6.append(concatenated_row)
######################
# x1 nodes 
F31= []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, d+1):# tau1 nodes
                  w = 0
                  if i==l:
                      w=1
                  temp_row.append(w)
            F31.append(temp_row)
##########
F32 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, d+1):# tau6 nodes
                  w = 0
                  if i==l:
                      w=-C
                  temp_row.append(w)
            F32.append(temp_row)
##########
F33 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F33.append(temp_row)

##########
F34 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F34.append(temp_row)
##########
F35 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F35.append(temp_row)
##########
F36 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F36.append(temp_row)
##########
F37 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F37.append(temp_row)
##########
F38 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F38.append(temp_row)
##########
F39 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F39.append(temp_row)
##########
F40 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F40.append(temp_row)
##########
for i in range(len(F31)):
    concatenated_row = F31[i] + F32[i] + F33[i] + F34[i]+F35[i] + F36[i] + F37[i] + F38[i] +F39[i] + F40[i]
    W6.append(concatenated_row)
######################
# x2 nodes as identyty map 
F41= []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F41.append(temp_row)
##########
F42 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F42.append(temp_row)
##########
F43 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F43.append(temp_row)

##########
F44 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F44.append(temp_row)
##########
F45 = []
for i in range(1, d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F45.append(temp_row)
##########
F46 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F46.append(temp_row)
##########
F47 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F47.append(temp_row)
##########
F48 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F48.append(temp_row)
##########
F49 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F49.append(temp_row)
##########
F50 = []
for i in range(1, d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  if i==j:
                      w=1
                  temp_row.append(w)
            F50.append(temp_row)
##########
for i in range(len(F41)):
    concatenated_row = F41[i] + F42[i] + F43[i] + F44[i]+F45[i] + F46[i] + F47[i] + F48[i] +F49[i] + F50[i]
    W6.append(concatenated_row)
#####################
# eta22 nodes as identity map 
F51= []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F51.append(temp_row)
##########
F52 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F52.append(temp_row)
##########
F53 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F53.append(temp_row)

##########
F54 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta22 nodes
                  w = 0
                  if i==l:
                      w=1
                  temp_row.append(w)
            F54.append(temp_row)
##########
F55 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta23 nodes
                  w = 0
                  temp_row.append(w)
            F55.append(temp_row)
##########
F56 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F56.append(temp_row)
##########
F57 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F57.append(temp_row)
##########
F58 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F58.append(temp_row)
##########
F59 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F59.append(temp_row)
##########
F60 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F60.append(temp_row)
##########
for i in range(len(F51)):
    concatenated_row = F51[i] + F52[i] + F53[i] + F54[i]+F55[i] + F56[i] + F57[i] + F58[i] +F59[i] + F60[i]
    W6.append(concatenated_row)
######################
# eta23 nodes as identity map 
F61= []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau1 nodes
                  w = 0
                  temp_row.append(w)
            F61.append(temp_row)
##########
F62 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, d+1):# tau6 nodes
                  w = 0
                  temp_row.append(w)
            F62.append(temp_row)
##########
F63 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# tau4 nodes
                  w = 0
                  temp_row.append(w)
            F63.append(temp_row)
##########
F64 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta22 nodes
                  w = 0
                  temp_row.append(w)
            F64.append(temp_row)
##########
F65 = []
for i in range(1, n+d+1):
            temp_row = []
            for l in range(1, n+d+1):# eta23 nodes
                  w = 0
                  if i==l:
                      w=1
                  temp_row.append(w)
            F65.append(temp_row)
##########
F66 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# alpha5,alpha6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F66.append(temp_row)
##########
F67 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# beta5,beta6 nodes
               for l in range(1, d+1):
                 for q in range(2):
                  w = 0
                  temp_row.append(w)
            F67.append(temp_row)
##########
F68 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi3 nodes for p'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F68.append(temp_row)
##########
F69 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):# psi4 nodes for q'ik nodes
               for l in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            F69.append(temp_row)
##########
F70 = []
for i in range(1, n+d+1):
            temp_row = []
            for j in range(1, d+1):# x2 nodes nodes
                  w = 0
                  temp_row.append(w)
            F70.append(temp_row)
##########
for i in range(len(F61)):
    concatenated_row = F61[i] + F62[i] + F63[i] + F64[i]+F65[i] + F66[i] + F67[i] + F68[i] +F69[i] + F70[i]
    W6.append(concatenated_row)
######################
# print("weight matrix for seventh layer/sixth hidden layer")
# print(W6)
#####################
# #Bias matrix for seventh layer/sixth hidden layer
B6 = []

# bias matrix tau4 nodes as identity map
for i in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)

# bias matrix for tau5 nodes

for i in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = d-i+1 
                temp_row.append(b)
            B6.append(temp_row)
            
# bias matrix for mu1(r_ik) nodes

for i in range(1, n+d+1):
  for k in range(1, n+d+1): 
            temp_row = []
            for l in range(1):
                b = V1[(i-1) * (n+d) + (k-1)] 
                temp_row.append(b)
            B6.append(temp_row)

# bias matrix for x1 nodes

for i in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)

# bias matrix for x2 nodes

for i in range(1, d+1): 
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)
# bias matrix for eta22 nodes

for i in range(1, n+d+1): 
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)
# bias matrix for eta23 nodes

for i in range(1, n+d+1): 
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)             
##################################
# print('Printing B6')
# for i in B6:
#     print(i)
##################################
L6 = []  # tau5, mu1(r_ik), x1 nodes
for i in range(len(W6)):
    temp_row = []
    L6_i_entry = np.maximum((np.dot(W6[i], L5)+B6[i]), 0)
    L6.append(L6_i_entry)
##################################
# print('Printing tau5, mu1(r_ik), x1 nodes for seventh layer/sixth hidden layer')
# for i in L6:
#     print(i)
############
# To construct weight matrix for eighth layer/seventh hidden layer is L7=W7*L6+B7
W7 = []
# tau4 as identity map 
G1 = []
for i in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# tau4
                w = 0
                if i == k:
                    w = 1
                temp_row.append(w)
        G1.append(temp_row)

##########
G2 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G2.append(temp_row)
##########
G3 = []
for l in range(1, d+1):
        temp_row = []
        for i in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G3.append(temp_row)
##########
G4 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):#x1
                    w = 0
                    temp_row.append(w)
        G4.append(temp_row)
##########
G5 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G5.append(temp_row)
##########
G6 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G6.append(temp_row)
##########
G7 = []
for i in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G7.append(temp_row)
##########
for i in range(len(G1)):
    concatenated_row = G1[i] + G2[i] + G3[i] + G4[i] + G5[i]  + G6[i] + G7[i] 
    W7.append(concatenated_row)
# #######################
# mu1(r_ik) nodes as identity map 
G8 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G8.append(temp_row)

##########
G9 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G9.append(temp_row)
##########
G10 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for j in range(1, n+d+1):
                    w = 0
                    if i == l and k==j:
                        w = 1
                    temp_row.append(w)
        G10.append(temp_row)
##########
G11 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x1
                    w = 0
                    temp_row.append(w)
        G11.append(temp_row)
##########
G12 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G12.append(temp_row)
##########
G13 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G13.append(temp_row)
##########
G14 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G14.append(temp_row)
##########
for i in range(len(G8)):
    concatenated_row = G8[i] + G9[i] + G10[i] + G11[i] + G12[i]  + G13[i] + G14[i] 
    W7.append(concatenated_row)
# print(W7)    
############################
# eta22 nodes as identity map 
G15 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G15.append(temp_row)

##########
G16 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G16.append(temp_row)
##########
G17 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G17.append(temp_row)
##########
G18 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x1
                    w = 0
                    temp_row.append(w)
        G18.append(temp_row)
##########
G19 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G19.append(temp_row)
##########
G20 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    if i == p:
                        w = 1
                    temp_row.append(w)
        G20.append(temp_row)
##########
G21 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G21.append(temp_row)
##########
for i in range(len(G15)):
    concatenated_row = G15[i] + G16[i] + G17[i] + G18[i] + G19[i]  + G20[i] + G21[i] 
    W7.append(concatenated_row)
############################
# eta23 nodes as identity map 
G22 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G22.append(temp_row)

##########
G23 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G23.append(temp_row)
##########
G24 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G24.append(temp_row)
##########
G25 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x1
                    w = 0
                    temp_row.append(w)
        G25.append(temp_row)
##########
G26 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G26.append(temp_row)
##########
G27 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G27.append(temp_row)
##########
G28 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    if i == p:
                        w = 1
                    temp_row.append(w)
        G28.append(temp_row)
##########
for i in range(len(G22)):
    concatenated_row = G22[i] + G23[i] + G24[i] + G25[i] + G26[i]  + G27[i] + G28[i] 
    W7.append(concatenated_row)
############################
# alpha7, alpha8 nodes 
G29 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G29.append(temp_row)

##########
G30 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G30.append(temp_row)
##########
G31 = []
for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G31.append(temp_row)
##########
G32 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            if j==k:
                w=1/eps
            temp_row.append(w)
        G32.append(temp_row)
##########
G33 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    if j==k:
                        w=-1/eps
                    temp_row.append(w)
        G33.append(temp_row)
##########
G34 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G34.append(temp_row)
##########
G35 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G35.append(temp_row)
##########
for i in range(len(G29)):
    concatenated_row = G29[i] + G30[i] + G31[i] + G32[i] + G33[i]  + G34[i] + G35[i] 
    W7.append(concatenated_row)
############################
# beta7, beta8 nodes 
G36 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G36.append(temp_row)

##########
G37 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G37.append(temp_row)
##########
G38 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G38.append(temp_row)
##########
G39 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            if j==k:
                w=-1/eps
            temp_row.append(w)
        G39.append(temp_row)
##########
G40 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    if j==k:
                        w=1/eps
                    temp_row.append(w)
        G40.append(temp_row)
##########
G41 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G41.append(temp_row)
##########
G42 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G42.append(temp_row)
##########
for i in range(len(G36)):
    concatenated_row = G36[i] + G37[i] + G38[i] + G39[i] + G40[i]  + G41[i] + G42[i] 
    W7.append(concatenated_row)
############################
# alpha9, alpha10 nodes 
G43 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G43.append(temp_row)

##########
G44 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G44.append(temp_row)
##########
G45 = []
for i in range(1, n+d+1):
  for k in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G45.append(temp_row)
##########
G46 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            if j==k:
                w=1/eps
            temp_row.append(w)
        G46.append(temp_row)
##########
G47 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G47.append(temp_row)
##########
G48 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G48.append(temp_row)
##########
G49 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G49.append(temp_row)
##########
for i in range(len(G43)):
    concatenated_row = G43[i] + G44[i] + G45[i] + G46[i] + G47[i]  + G48[i] + G49[i] 
    W7.append(concatenated_row)
############################
# beta9, beta10 nodes 
G50 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G50.append(temp_row)
##########
G51 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G51.append(temp_row)
##########
G52 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G52.append(temp_row)
##########
G53 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            if j==k:
                w=-1/eps
            temp_row.append(w)
        G53.append(temp_row)
##########
G54 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G54.append(temp_row)
##########
G55 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G55.append(temp_row)
##########
G56 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G56.append(temp_row)
##########
for i in range(len(G50)):
    concatenated_row = G50[i] + G51[i] + G52[i] + G53[i] + G54[i]  + G55[i] + G56[i] 
    W7.append(concatenated_row)
############################
# alpha11, alpha12 nodes 
G57 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G57.append(temp_row)

##########
G58 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G58.append(temp_row)
##########
G59 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G59.append(temp_row)
##########
G60 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x1
            w = 0
            if j==i:
                w=1/eps
            temp_row.append(w)
        G60.append(temp_row)
##########
G61 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G61.append(temp_row)
##########
G62 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G62.append(temp_row)
##########
G63 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G63.append(temp_row)
##########
for i in range(len(G57)):
    concatenated_row = G57[i] + G58[i] + G59[i] + G60[i] + G61[i]  + G62[i] + G63[i] 
    W7.append(concatenated_row)
############################
# beta11, beta12 nodes 
G64 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G64.append(temp_row)
##########
G65 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G65.append(temp_row)
##########
G66 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G66.append(temp_row)
##########
G67 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x1
            w = 0
            if j==i:
                w=-1/eps
            temp_row.append(w)
        G67.append(temp_row)
##########
G68 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G68.append(temp_row)
##########
G69 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G69.append(temp_row)
##########
G70 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G70.append(temp_row)
##########
for i in range(len(G64)):
    concatenated_row = G64[i] + G65[i] + G66[i] + G67[i] + G68[i]  + G69[i] + G70[i] 
    W7.append(concatenated_row)
############################
# alpha15, alpha16 nodes 
G71 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G71.append(temp_row)

##########
G72 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G72.append(temp_row)
##########
G73 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G73.append(temp_row)
##########
G74 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G74.append(temp_row)
##########
G75 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    if j==k:
                        w=1/eps
                    temp_row.append(w)
        G75.append(temp_row)
##########
G76 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G76.append(temp_row)
##########
G77 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G77.append(temp_row)
##########
for i in range(len(G71)):
    concatenated_row = G71[i] + G72[i] + G73[i] + G74[i] + G75[i]  + G76[i] + G77[i] 
    W7.append(concatenated_row)
############################
# beta15, beta16 nodes 
G78 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G78.append(temp_row)
##########
G79 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G79.append(temp_row)
##########
G80 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G80.append(temp_row)
##########
G81 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G81.append(temp_row)
##########
G82 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    if j==k:
                        w=-1/eps
                    temp_row.append(w)
        G82.append(temp_row)
##########
G83 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G83.append(temp_row)
##########
G84 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G84.append(temp_row)
##########
for i in range(len(G78)):
    concatenated_row = G78[i] + G79[i] + G80[i] + G81[i] + G82[i]  + G83[i] + G84[i] 
    W7.append(concatenated_row)
############################
# alpha13, alpha14 nodes 
G85 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G85.append(temp_row)

##########
G86 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G86.append(temp_row)
##########
G87 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G87.append(temp_row)
##########
G88 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G88.append(temp_row)
##########
G89 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x2
                    w = 0
                    if j==i:
                        w=1/eps
                    temp_row.append(w)
        G89.append(temp_row)
##########
G90 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G90.append(temp_row)
##########
G91 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G91.append(temp_row)
##########
for i in range(len(G85)):
    concatenated_row = G85[i] + G86[i] + G87[i] + G88[i] + G89[i]  + G90[i] + G91[i] 
    W7.append(concatenated_row)
############################
# beta13, beta14 nodes 
G92 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G92.append(temp_row)
##########
G93 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G93.append(temp_row)
##########
G94 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G94.append(temp_row)
##########
G95 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G95.append(temp_row)
##########
G96 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x2
                    w = 0
                    if j==i:
                        w=-1/eps
                    temp_row.append(w)
        G96.append(temp_row)
##########
G97 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G97.append(temp_row)
##########
G98 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G98.append(temp_row)
##########
for i in range(len(G92)):
    concatenated_row = G92[i] +G93[i] + G94[i] + G95[i] + G96[i] + G97[i]  + G98[i] 
    W7.append(concatenated_row)
############################
# alpha17, alpha18 nodes 
G99 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G99.append(temp_row)
##########
G100 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):# tau5
              w = 0
              if k==i:
                  w=-1/eps
              temp_row.append(w)
        G100.append(temp_row)
##########
G101 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G101.append(temp_row)
##########
G102 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G102.append(temp_row)
##########
G103 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G103.append(temp_row)
##########
G104 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G104.append(temp_row)
##########
G105 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G105.append(temp_row)
##########
for i in range(len(G99)):
    concatenated_row = G99[i] + G100[i] + G101[i] + G102[i] + G103[i]  + G104[i] + G105[i] 
    W7.append(concatenated_row)
############################
# beta17, beta18 nodes 
G106 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G106.append(temp_row)
##########
G107 = []
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):# tau5
              w = 0
              if k==i:
                  w=1/eps
              temp_row.append(w)
        G107.append(temp_row)
##########
G108 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for i in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G108.append(temp_row)
##########
G109 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G109.append(temp_row)
##########
G110 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G110.append(temp_row)
##########
G111 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G111.append(temp_row)
##########
G112 = []
for k in range(1, d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G112.append(temp_row)
##########
for i in range(len(G106)):
    concatenated_row = G106[i] +G107[i] + G108[i] + G109[i] + G110[i] + G111[i]  + G112[i] 
    W7.append(concatenated_row)
############################
# alpha19, alpha20 nodes 
G113 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G113.append(temp_row)
##########
G114 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G114.append(temp_row)
##########
G115 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G115.append(temp_row)
##########
G116 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G116.append(temp_row)
##########
G117 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G117.append(temp_row)
##########
G118 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G118.append(temp_row)
##########
G119 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G119.append(temp_row)
##########
for i in range(len(G113)):
    concatenated_row = G113[i] + G114[i] + G115[i] + G116[i] + G117[i]  + G118[i] + G119[i] 
    W7.append(concatenated_row)
############################
# beta19, beta20 nodes 
G120 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, d+1):# tau4
                w=0
                temp_row.append(w)
        G120.append(temp_row)
##########
G121 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):# tau5
              w = 0
              temp_row.append(w)
        G121.append(temp_row)
##########
G122 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):#  mu1(r_ik) nodes
          for k in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
        G122.append(temp_row)
##########
G123 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x1
            w = 0
            temp_row.append(w)
        G123.append(temp_row)
##########
G124 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1, d+1):#x2
                    w = 0
                    temp_row.append(w)
        G124.append(temp_row)
##########
G125 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta22 nodes
                    w = 0
                    temp_row.append(w)
        G125.append(temp_row)
##########
G126 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1, n+d+1):#eta23 nodes
                    w = 0
                    temp_row.append(w)
        G126.append(temp_row)
##########
for i in range(len(G120)):
    concatenated_row = G120[i] +G121[i] + G122[i] + G123[i] + G124[i] + G125[i]  + G126[i] 
    W7.append(concatenated_row)
############################
# print("weight matrix for eighth layer/seventh hidden layer")
# print(W7)
#####################
#Bias matrix for eighth layer/seventh hidden layer

B7 = []

# bias matrix for tau4 nodes

for i in range(1, d+1):
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B7.append(temp_row)
        
# bias matrix for mu1(r_ik) nodes 

for i in range(1, n+d+1):
      for j in range(1, n+d+1):
        temp_row = []
        for k in range(1):
            b = 0
            temp_row.append(b)
        B7.append(temp_row)

# bias matrix for eta22 nodes 

for i in range(1, n+d+1):
        temp_row = []
        for k in range(1):
            b=0
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for eta23 nodes 

for i in range(1, n+d+1):
        temp_row = []
        for k in range(1):
            b=0
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for alpha7, alpha8 nodes 
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1
            temp_row.append(b)
        B7.append(temp_row)  
# bias matrix for beta7, beta8 nodes 
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for alpha9, alpha10 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(-i/eps)
            else:
                b=(-i/eps)
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta9, beta10 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(i/eps)
            else:
                b=(i/eps)
            temp_row.append(b)
        B7.append(temp_row)  
# bias matrix for alpha11, alpha12 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(-i/eps)
            else:
                b=(-i/eps)
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta11, beta12 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(i/eps)
            else:
                b=(i/eps)
            temp_row.append(b)
        B7.append(temp_row) 
# bias matrix for alpha15, alpha16 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(-i/eps)
            else:
                b=(-i/eps)
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta15, beta16 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b=0
            if q==0:
                b=1+(i/eps)
            else:
                b=(i/eps)
            temp_row.append(b)
        B7.append(temp_row) 
# bias matrix for alpha13,alpha14 nodes 
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
            b=0
            if q==0:
                b=1+(-k/eps)
            else:
                b=(-k/eps)
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta13, beta14 nodes 
for k in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
            b=0
            if q==0:
                b=1+(k/eps)
            else:
                b=(k/eps)
            temp_row.append(b)
        B7.append(temp_row) 
# bias matrix for alpha17,alpha18 nodes 
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1):
            b=0
            if q==0:
                b=1+((j-1)/eps)
            else:
                b=(j-1)/eps
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta17, beta18 nodes 
for j in range(1, d+1):
  for i in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1):
            b=0
            if q==0:
                b=1+((-j+1)/eps)
            else:
                b=(-j+1)/eps
            temp_row.append(b)
        B7.append(temp_row)  
# bias matrix for alpha19,alpha20 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1):
            b=0
            if q==0:
                b=1+((i-j-n)/eps)
            else:
                b=(i-j-n)/eps
            temp_row.append(b)
        B7.append(temp_row)
# bias matrix for beta19, beta20 nodes 
for i in range(1, n+d+1):
  for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for p in range(1):
            b=0
            if q==0:
                b=1+((j-i+n)/eps)
            else:
                b=(j-i+n)/eps
            temp_row.append(b)
        B7.append(temp_row)         
# print('Printing B7')
# for i in B7:
#     print(i)
##################################
L7 = []  # alpha7,8, beta7,8, alpha9,10 beta9,10, alpha11,12, beta 11,12, alpha13,14, beta 13,14 alpha15,16, beta 15,16, alpha17,18, beta 17,18, alpha19,20, beta 19,20 nodes
for i in range(len(W7)):
    temp_row = []
    L7_i_entry = np.maximum((np.dot(W7[i], L6)+B7[i]), 0)
    L7.append(L7_i_entry)
##################################
# print('Printing alpha7,8, beta7,8, alpha9,10 beta9,10, alpha11,12, beta 11,12, alpha13,14, beta 13,14 alpha15,16, beta 15,16, alpha17,18, beta 17,18, alpha19,20, beta 19,20 nodes for eighth layer/seventh hidden layer')
# for i in L7:
#     print(i)
#####################
# To construct weight matrix for ninth layer/eighth hidden layer is L7=W7*L6+B7
W8 = []
# mu1(r_ik) nodes as identity map
H1 = []
for i in range(1, n+d+1):
     for k in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H1.append(temp_row)
##########
H2 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for j in range(1, n+d+1):
                    w = 0
                    if i==l and k==j:
                        w=1
                    temp_row.append(w)
          H2.append(temp_row)
##########
H3 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for p in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H3.append(temp_row)
##########
H5 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for p in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H5.append(temp_row)
##########
H6 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H6.append(temp_row)
##########
H7 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H7.append(temp_row)
##########
H8 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H8.append(temp_row)
##########
H9 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H9.append(temp_row)
##########
H10 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H10.append(temp_row)
##########
H11 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H11.append(temp_row)
##########
H12 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H12.append(temp_row)
##########
H13 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H13.append(temp_row)
##########
H14 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H14.append(temp_row)
##########
H15 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
          temp_row = []          
          for k in range(1, n+d+1):#beta13, beta14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H15.append(temp_row)
##########
H16 = []
for l in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha17, alpha18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H16.append(temp_row)
##########
H17 = []
for l in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#beta17, beta18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H17.append(temp_row)
##########
H18 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H18.append(temp_row)
##########
H19 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H19.append(temp_row)
##########
for i in range(len(H1)):
    concatenated_row = H1[i]+ H2[i]+ H3[i]+ H5[i] + H6[i]+ H7[i] + H8[i] + H9[i]+ H10[i]+H11[i]+ H12[i]+ H13[i]+ H14[i] + H15[i] + H16[i]+ H17[i] + H18[i] + H19[i] 
    W8.append(concatenated_row)
######################
# eta22 nodes as identity map
H20 = []
for i in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H20.append(temp_row)
##########
H21 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H21.append(temp_row)
##########
H22 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    if i==l:
                        w=1
                    temp_row.append(w)
          H22.append(temp_row)
##########
H23 = []
for i in range(1, n+d+1):
          temp_row = []          
          for p in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H23.append(temp_row)
##########
H24 = []
for i in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H24.append(temp_row)
##########
H25 = []
for i in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H25.append(temp_row)
##########
H26 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H26.append(temp_row)
##########
H27 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H27.append(temp_row)
##########
H28 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H28.append(temp_row)
##########
H29 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H29.append(temp_row)
##########
H30 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H30.append(temp_row)
##########
H31 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H31.append(temp_row)
##########
H32 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H32.append(temp_row)
##########
H33 = []
for i in range(1, n+d+1):
          temp_row = []          
          for k in range(1, n+d+1):#beta13, beta14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H33.append(temp_row)
##########
H34 = []
for l in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha17, alpha18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H34.append(temp_row)
##########
H35 = []
for l in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#beta17, beta18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H35.append(temp_row)
##########
H36 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H36.append(temp_row)
##########
H37 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H37.append(temp_row)
##########
for i in range(len(H20)):
    concatenated_row = H20[i]+ H21[i]+ H22[i]+ H23[i] + H24[i]+ H25[i] + H26[i] + H27[i]+ H28[i]+H29[i]+ H30[i]+ H31[i]+ H32[i] + H33[i] + H34[i]+ H35[i] + H36[i] + H37[i] 
    W8.append(concatenated_row)
######################
# eta23 nodes as identity map
H38 = []
for i in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H38.append(temp_row)
##########
H39 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for j in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H39.append(temp_row)
##########
H40 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H40.append(temp_row)
##########
H41 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta23 nodes
                    w = 0
                    if i==l:
                        w=1
                    temp_row.append(w)
          H41.append(temp_row)
##########
H42 = []
for i in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H42.append(temp_row)
##########
H43 = []
for i in range(1, n+d+1):
          temp_row = []
          for j in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H43.append(temp_row)
##########
H44 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H44.append(temp_row)
##########
H45 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H45.append(temp_row)
##########
H46 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H46.append(temp_row)
##########
H47 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H47.append(temp_row)
##########
H48 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H48.append(temp_row)
##########
H49 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H49.append(temp_row)
##########
H50 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H50.append(temp_row)
##########
H51 = []
for i in range(1, n+d+1):
          temp_row = []          
          for k in range(1, n+d+1):#beta13, beta14
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H51.append(temp_row)
##########
H52 = []
for l in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#alpha17, alpha18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H52.append(temp_row)
##########
H53 = []
for l in range(1, n+d+1):
          temp_row = []          
          for j in range(1, d+1):#beta17, beta18
            for i in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H53.append(temp_row)
##########
H54 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H54.append(temp_row)
##########
H55 = []
for i in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for j in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H55.append(temp_row)
##########
for i in range(len(H38)):
    concatenated_row = H38[i]+ H39[i]+ H40[i]+ H41[i] + H42[i]+ H43[i] + H44[i] + H45[i]+ H46[i]+H47[i]+ H48[i]+ H49[i]+ H50[i] + H51[i] + H52[i]+ H53[i] + H54[i] + H55[i] 
    W8.append(concatenated_row)
###################
# zeta1 nodes
H56 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H56.append(temp_row)
##########
H57 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for p in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H57.append(temp_row)
##########
H58 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H58.append(temp_row)
##########
H59 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H59.append(temp_row)
##########
H60 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    if j==l:
                        if q==0:
                          w=-1
                        else:
                            w=1
                    temp_row.append(w)
          H60.append(temp_row)
##########
H61 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    if j==l:
                        if q==0:
                          w=-1
                        else:
                            w=1
                    temp_row.append(w)
          H61.append(temp_row)
##########
H62 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if i==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H62.append(temp_row)
##########
H63 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if i==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H63.append(temp_row)
##########
H64 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H64.append(temp_row)
##########
H65 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H65.append(temp_row)
##########
H66 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H66.append(temp_row)
##########
H67 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H67.append(temp_row)
##########
H68 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if k==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H68.append(temp_row)
##########
H69 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta13, beta14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if k==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H69.append(temp_row)
##########
H70 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha17, alpha18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H70.append(temp_row)
##########
H71 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, d+1):#beta17, beta18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H71.append(temp_row)
##########
H72 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H72.append(temp_row)
##########
H73 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H73.append(temp_row)
##########
for i in range(len(H56)):
    concatenated_row = H56[i]+ H57[i]+ H58[i]+ H59[i] + H60[i]+ H61[i] + H62[i] + H63[i]+ H64[i]+H65[i]+ H66[i]+ H67[i]+ H68[i] + H69[i] + H70[i]+ H71[i] + H72[i] + H73[i] 
    W8.append(concatenated_row)
###################
# zeta prime nodes
H156 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H156.append(temp_row)
##########
H157 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for p in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H157.append(temp_row)
##########
H158 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H158.append(temp_row)
##########
H159 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H159.append(temp_row)
##########
H160 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    if j==l:
                        if q==0:
                          w=-1
                        else:
                            w=1
                    temp_row.append(w)
          H160.append(temp_row)
##########
H161 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    if j==l:
                        if q==0:
                          w=-1
                        else:
                            w=1
                    temp_row.append(w)
          H161.append(temp_row)
##########
H162 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H162.append(temp_row)
##########
H163 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H163.append(temp_row)
##########
H164 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if k==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H164.append(temp_row)
##########
H165 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if k==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H165.append(temp_row)
##########
H166 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if i==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H166.append(temp_row)
##########
H167 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if i==l and j==p:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H167.append(temp_row)
##########
H168 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H168.append(temp_row)
##########
H169 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta13, beta14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H169.append(temp_row)
##########
H170 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha17, alpha18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H170.append(temp_row)
##########
H171 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, d+1):#beta17, beta18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H171.append(temp_row)
##########
H172 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H172.append(temp_row)
##########
H173 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, n+d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H173.append(temp_row)
##########
for i in range(len(H156)):
    concatenated_row = H156[i]+ H157[i]+ H158[i]+ H159[i] + H160[i]+ H161[i] + H162[i] + H163[i]+ H164[i]+H165[i]+ H166[i]+ H167[i]+ H168[i] + H169[i] + H170[i]+ H171[i] + H172[i] + H173[i] 
    W8.append(concatenated_row)
###################
# Tau6 nodes
H74 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):# tau4
            w = 0
            if i==l:
                  w=1
            temp_row.append(w)
          H74.append(temp_row)
##########
H75 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for p in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H75.append(temp_row)
##########
H76 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H76.append(temp_row)
##########
H77 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H77.append(temp_row)
##########
H78 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H78.append(temp_row)
##########
H79 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H79.append(temp_row)
##########
H80 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H80.append(temp_row)
##########
H81 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H81.append(temp_row)
##########
H82 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H82.append(temp_row)
##########
H83 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H83.append(temp_row)
##########
H84 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H84.append(temp_row)
##########
H85 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H85.append(temp_row)
##########
H86 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H86.append(temp_row)
##########
H87 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta13, beta14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H87.append(temp_row)
##########
H88 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha17, alpha18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if j==l and i==p:
                        if q==0:
                          w=C
                        else:
                            w=-C
                    temp_row.append(w)
          H88.append(temp_row)
##########
H89 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#beta17, beta18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if j==l and i==p:
                        if q==0:
                          w=C
                        else:
                            w=-C
                    temp_row.append(w)
          H89.append(temp_row)
##########
H90 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H90.append(temp_row)
##########
H91 = []
for j in range(1, d+1):
      for i in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H91.append(temp_row)
##########
for i in range(len(H74)):
    concatenated_row = H74[i]+ H75[i]+ H76[i]+ H77[i] + H78[i]+ H79[i] + H80[i] + H81[i]+ H82[i]+H83[i]+ H84[i]+ H85[i]+ H86[i] + H87[i] + H88[i]+ H89[i] + H90[i] + H91[i] 
    W8.append(concatenated_row)
###################
# Tau7 nodes
H92 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):# tau4
            w = 0
            temp_row.append(w)
          H92.append(temp_row)
##########
H93 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# mu1(r_ik) nodes
               for p in range(1, n+d+1):
                    w = 0
                    temp_row.append(w)
          H93.append(temp_row)
##########
H94 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta22
                    w = 0
                    temp_row.append(w)
          H94.append(temp_row)
##########
H95 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):# eta23 nodes
                    w = 0
                    temp_row.append(w)
          H95.append(temp_row)
##########
H96 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha7, alpha8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H96.append(temp_row)
##########
H97 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []
          for l in range(1, d+1):#beta7, beta8
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H97.append(temp_row)
##########
H98 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha9, alpha10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H98.append(temp_row)
##########
H99 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta9, beta10
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H99.append(temp_row)
##########
H100 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha11, alpha12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H100.append(temp_row)
##########
H101 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta11, beta12
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H101.append(temp_row)
##########
H102 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha15, alpha16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H102.append(temp_row)
##########
H103 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta15, beta16
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H103.append(temp_row)
##########
H104 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha13, alpha14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H104.append(temp_row)
##########
H105 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta13, beta14
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H105.append(temp_row)
##########
H106 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#alpha17, alpha18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H106.append(temp_row)
##########
H107 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, d+1):#beta17, beta18
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    temp_row.append(w)
          H107.append(temp_row)
##########
H108 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#alpha19, alpha20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if j==p and i==l:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H108.append(temp_row)
##########
H109 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
          temp_row = []          
          for l in range(1, n+d+1):#beta19, beta20
            for p in range(1, d+1):
              for q in range(2):
                    w = 0
                    if j==p and i==l:
                        if q==0:
                          w=1
                        else:
                            w=-1
                    temp_row.append(w)
          H109.append(temp_row)
##########
for i in range(len(H92)):
    concatenated_row = H92[i]+ H93[i]+ H94[i]+ H95[i] + H96[i]+ H97[i] + H98[i] + H99[i]+ H100[i]+H101[i]+ H102[i]+ H103[i]+ H104[i] + H105[i] + H106[i]+ H107[i] + H108[i] + H109[i] 
    W8.append(concatenated_row)
###################
# print("weight matrix for ninth layer/eighth hidden layer")
# print(W8)
#####################
# #Bias matrix for ninth layer/eighth hidden layer
B8 = []

# bias matrix for mu1(r_ik) nodes

for i in range(1, n+d+1):
     for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 0
          temp_row.append(b)
      B8.append(temp_row)
# bias matrix for eta22 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 0
          temp_row.append(b)
      B8.append(temp_row)
# bias matrix for eta23 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 0
          temp_row.append(b)
      B8.append(temp_row)
      
# bias matrix for zeta1 nodes 
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1):
            b=-2
            temp_row.append(b)
        B8.append(temp_row) 
      
# bias matrix for zeta prime nodes 
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1):
            b=-2
            temp_row.append(b)
        B8.append(temp_row)        
# bias matrix for tau6 nodes 

for i in range(1, d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1):
            b=-2*C
            temp_row.append(b)
        B8.append(temp_row)
# bias matrix for tau7 nodes 

for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1):
            b=-1
            temp_row.append(b)
        B8.append(temp_row)             
# print('Printing B8')
# for i in B8:
#     print(i)
#####################################
##################################
L8 = []  # zeta1,zeta prime, tau6, tau7 nodes
for i in range(len(W8)):
    temp_row = []
    L8_i_entry = np.maximum((np.dot(W8[i], L7)+B8[i]), 0)
    L8.append(L8_i_entry)
############################
# print('Printing zeta1, tau6, tau7 nodes for ninth layer/eighth hidden layer')
# for i in L8:
#     print(i)
############################
# To construct weight matrix for tenth layer/ninth hidden layer is L9=W9*L8+B9
W9 = []
# mu1(r_ik) nodes as identity map 
I1 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
              w = 0
              if i==l and k==p:
                  w=1
              temp_row.append(w)
        I1.append(temp_row)
##########
I2 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              temp_row.append(w)
        I2.append(temp_row)
##########
I3 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):# eta23 nodes
              w = 0
              temp_row.append(w)
        I3.append(temp_row)
##########
I4 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I4.append(temp_row)
##########
I74 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta prime nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I74.append(temp_row)
##########
I5 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):#tau6 nodes
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I5.append(temp_row)
##########
I6 = []
for i in range(1, n+d+1):
      for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I6.append(temp_row)
##########
for i in range(len(I1)):
    concatenated_row = I1[i] + I2[i]+ I3[i]+I4[i]+I74[i] + I5[i]+ I6[i]
    W9.append(concatenated_row)    
#######################
# eta22 nodes as identity map 
I7 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              if i==l:
                  w=1
              temp_row.append(w)
        I8.append(temp_row)
##########
I9 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):# eta23 nodes
              w = 0
              temp_row.append(w)
        I9.append(temp_row)
##########
I10 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I10.append(temp_row)
##########
I70 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta prime nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I70.append(temp_row)
##########
I11 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):#tau6 nodes
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I11.append(temp_row)
##########
I12 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I12.append(temp_row)
##########
for i in range(len(I7)):
    concatenated_row = I7[i]+I8[i]+I9[i]+I10[i]+I70[i]+I11[i]+I12[i]
    W9.append(concatenated_row)    
####################### 
# eta23 nodes as identity map 
I13 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I13.append(temp_row)
##########
I14 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              temp_row.append(w)
        I14.append(temp_row)
##########
I15 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta23 nodes
              w = 0
              if i==l:
                  w=1
              temp_row.append(w)
        I15.append(temp_row)
##########
I16 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I16.append(temp_row)
##########
I76 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta prime nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I76.append(temp_row)
##########
I17 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, d+1):#tau6 nodes
              for j in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I17.append(temp_row)
##########
I18 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I18.append(temp_row)
##########
for i in range(len(I13)):
    concatenated_row = I13[i]+I14[i]+I15[i]+I16[i]+I76[i]+I17[i]+I18[i]
    W9.append(concatenated_row)    
#######################
# tau7 nodes as identity map 
I19 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I19.append(temp_row)
##########
I20 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              temp_row.append(w)
        I20.append(temp_row)
##########
I21 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta23 nodes
              w = 0
              temp_row.append(w)
        I21.append(temp_row)
##########
I22 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I22.append(temp_row)
##########
I77 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta prime nodes
            for p in range(1, n+d+1):
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I77.append(temp_row)
##########
I23 = []
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1, d+1):#tau6 nodes
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I23.append(temp_row)
##########
I24 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
            for k in range(1, d+1):
                w = 0
                if i==l and j==k:
                    w=1
                temp_row.append(w)
        I24.append(temp_row)
##########
for i in range(len(I19)):
    concatenated_row = I19[i]+I20[i]+I21[i]+I22[i]+I77[i]+I23[i]+I24[i]
    W9.append(concatenated_row)    
# ####################### 
# x3 nodes 
I25 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I25.append(temp_row)
##########
I26 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              temp_row.append(w)
        I26.append(temp_row)
##########
I27 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta23 nodes
              w = 0
              temp_row.append(w)
        I27.append(temp_row)
##########
I28 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I28.append(temp_row)
##########
I78 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):#  zeta prime nodes
            for p in range(1, n+d+1):
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I78.append(temp_row)
##########
I29 = []
for k in range(1, d+1):
        temp_row = []
        for j in range(1, d+1):#tau6 nodes
              for i in range(1, d+1):
                  w = 0
                  if j==k:
                      w=1
                  temp_row.append(w)
        I29.append(temp_row)
##########
I30 = []
for i in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
            for k in range(1, d+1):
                w = 0
                temp_row.append(w)
        I30.append(temp_row)
##########
for i in range(len(I25)):
    concatenated_row = I25[i]+I26[i]+I27[i]+I28[i]+I78[i]+I29[i]+I30[i]
    W9.append(concatenated_row)    
#######################
# zeta2 nodes 
I31 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) nodes
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        I31.append(temp_row)
##########
I32 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22 nodes
              w = 0
              temp_row.append(w)
        I32.append(temp_row)
##########
I33 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta23 nodes
              w = 0
              temp_row.append(w)
        I33.append(temp_row)
##########
I34 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta1 nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                 w = 0
                 if i==l and k==p:
                     w=1
                 temp_row.append(w)
        I34.append(temp_row)
##########
I80 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta prime nodes
            for p in range(1, n+d+1):
              for j in range(1, d+1):
                 w = 0
                 if i==l and k==p:
                     w=1
                 temp_row.append(w)
        I80.append(temp_row)
##########
I35 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for j in range(1, d+1):#tau6 nodes
              for l in range(1, d+1):
                  w = 0
                  temp_row.append(w)
        I35.append(temp_row)
##########
I36 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7 nodes
            for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I36.append(temp_row)
##########
for i in range(len(I31)):
    concatenated_row = I31[i]+I32[i]+I33[i]+I34[i]+I80[i]+I35[i]+I36[i]
    W9.append(concatenated_row)    
#######################
# print("weight matrix for tenth layer/ninth hidden layer")
# print(W9)
#####################
# #Bias matrix for tenth layer/ninth hidden layer
#B9 = [] is a zero matrix
##################################
L9 = []  # x3, zeta2, tau7 nodes
for i in range(len(W9)):
    temp_row = []
    L9_i_entry = np.maximum((np.dot(W9[i], L8)), 0)
    L9.append(L9_i_entry)
###################################
# print('Printing x3, zeta2 nodes for tenth layer/ninth hidden layer')
# for i in L9:
#     print(i)
##################################
# To construct weight matrix for eleventh layer/tenth hidden layer is L10=W10*L9+B10
W10 = []
# eta22 nodes as identity map 
I37 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) 
          for k in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        I37.append(temp_row)
##########
I38 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta22
                w = 0
                if i==k:
                    w=1
                temp_row.append(w)
        I38.append(temp_row)
##########
I39 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        I39.append(temp_row)
##########
I40 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7
             for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I40.append(temp_row)
##########
I41 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#x3
             w = 0
             temp_row.append(w)
        I41.append(temp_row)
##########
I42 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta2 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        I42.append(temp_row)
##########
for i in range(len(I37)):
    concatenated_row = I37[i] + I38[i] +I39[i] + I40[i]+I41[i] + I42[i]
    W10.append(concatenated_row)    
######################
# eta23 nodes as identity map 
I43 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) 
          for k in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I43.append(temp_row)
##########
I44 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta22
                w = 0
                temp_row.append(w)
        I44.append(temp_row)
##########
I45 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
              w = 0
              if i==k:
                  w=1
              temp_row.append(w)
        I45.append(temp_row)
##########
I46 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        I46.append(temp_row)
##########
I47 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#x3
              w = 0
              temp_row.append(w)
        I47.append(temp_row)
##########
I48 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta2 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        I48.append(temp_row)
##########
for i in range(len(I43)):
    concatenated_row = I43[i] + I44[i] +I45[i] + I46[i]+I47[i] + I48[i]
    W10.append(concatenated_row)    
######################
# tau8 nodes 
I49 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) 
          for k in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I49.append(temp_row)
##########
I50 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta22
                w = 0
                temp_row.append(w)
        I50.append(temp_row)
##########
I51 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
              w = 0
              temp_row.append(w)
        I51.append(temp_row)
##########
I52 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7
              for k in range(1, d+1):
                w = 0
                if i==l and j==k:
                    w=C
                temp_row.append(w)
        I52.append(temp_row)
##########
I53 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):#x3
              w = 0
              if j==k:
                  w=1
              temp_row.append(w)
        I53.append(temp_row)
##########
I54 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta2 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        I54.append(temp_row)
##########
for i in range(len(I49)):
    concatenated_row = I49[i] + I50[i] +I51[i] + I52[i]+I53[i] + I54[i]
    W10.append(concatenated_row)    
######################
# zeta3 nodes 
I55 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) 
          for j in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I55.append(temp_row)
##########
I56 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):## eta22
                w = 0
                temp_row.append(w)
        I56.append(temp_row)
##########
I57 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
              w = 0
              temp_row.append(w)
        I57.append(temp_row)
##########
I58 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7
              for k in range(1, d+1):
                w = 0
                temp_row.append(w)
        I58.append(temp_row)
##########
I59 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#x3
              w = 0
              temp_row.append(w)
        I59.append(temp_row)
##########
I60 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta2 nodes
          for k in range(1, n+d+1):
                w = 0
                if i==l and j==k:
                    w=C
                temp_row.append(w)
        I60.append(temp_row)
##########
for i in range(len(I55)):
    concatenated_row = I55[i] + I56[i] +I57[i] + I58[i]+I59[i] + I60[i]
    W10.append(concatenated_row)    
######################
# zeta4 nodes 
I61 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# mu1(r_ik) 
          for j in range(1, n+d+1):
              w = 0
              if i==l and k==j:
                  w=1
              temp_row.append(w)
        I61.append(temp_row)
##########
I62 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):## eta22
                w = 0
                temp_row.append(w)
        I62.append(temp_row)
##########
I63 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
              w = 0
              temp_row.append(w)
        I63.append(temp_row)
##########
I64 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau7
              for k in range(1, d+1):
                w = 0
                temp_row.append(w)
        I64.append(temp_row)
##########
I65 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):#x3
              w = 0
              temp_row.append(w)
        I65.append(temp_row)
##########
I66 = []
for i in range(1, n+d+1):
    for j in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta2 nodes
          for k in range(1, n+d+1):
                w = 0
                if i==l and j==k:
                    w=-C
                temp_row.append(w)
        I66.append(temp_row)
##########
for i in range(len(I61)):
    concatenated_row = I61[i] + I62[i] +I63[i] + I64[i]+I65[i] + I66[i]
    W10.append(concatenated_row)    
# ######################
# print("weight matrix for tenth layer/ninth hidden layer")
# print(W10)
#####################

# #Bias matrix for tenth layer/ninth hidden layer

B10=[]
# bias matrix for eta22 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 0
          temp_row.append(b)
      B10.append(temp_row)
# bias matrix for eta23 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 0
          temp_row.append(b)
      B10.append(temp_row)
      
# bias matrix for tau8 nodes 
for i in range(1, n+d+1):
      for j in range(1, d+1):
        temp_row = []
        for l in range(1):
            b=-C
            temp_row.append(b)
        B10.append(temp_row) 
        
# bias matrix for zeta3 nodes 

for i in range(1, n+d+1):
      for j in range(1, n+d+1):
        temp_row = []
        for l in range(1):
            b=1-C
            temp_row.append(b)
        B10.append(temp_row)
# bias matrix for zeta4 nodes 
for i in range(1, n+d+1):
      for j in range(1, n+d+1):
        temp_row = []
        for l in range(1):
            b=0
            temp_row.append(b)
        B10.append(temp_row)            
# print('Printing B10')
# for i in B10:
#     print(i)     
################################## 
L10 = []  # tau8, zeta3, zeta4
for i in range(len(W10)):
    temp_row = []
    L10_i_entry = np.maximum((np.dot(W10[i], L9)+B10[i]), 0)
    L10.append(L10_i_entry)
#################################
# print('Printing tau8, zeta3, zeta4 nodes for eleventh layer/tenth hidden layer')
# for i in L10:
#     print(i)
############
# To construct weight matrix for twelfth layer/eleventh hidden layer is L11=W11*L10+B11
W11 = []
# eta22 nodes as identity map 
J1 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             if i==l:
                 w=1
             temp_row.append(w)
        J1.append(temp_row)
##########
J2 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        J2.append(temp_row)
##########
J3 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#tau8
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        J3.append(temp_row)
##########
J4 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#zeta3
          for j in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        J4.append(temp_row)
##########
J5 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta4 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J5.append(temp_row)
##########
for i in range(len(J1)):
    concatenated_row = J1[i] + J2[i] +J3[i] + J4[i]+J5[i] 
    W11.append(concatenated_row)    
######################
## eta23 nodes as identity map 
J6 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             temp_row.append(w)
        J6.append(temp_row)
##########
J7 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
             w = 0
             if i==k:
                 w=1
             temp_row.append(w)
        J7.append(temp_row)
##########
J8 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#tau8
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        J8.append(temp_row)
##########
J9 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#zeta3
          for j in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        J9.append(temp_row)
##########
J10 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta4 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J10.append(temp_row)
##########
for i in range(len(J6)):
    concatenated_row = J6[i] + J7[i] +J8[i] + J9[i]+J10[i] 
    W11.append(concatenated_row)    
######################
## nu1 nodes as identity map 
J11 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             temp_row.append(w)
        J11.append(temp_row)
##########
J12 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for j in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        J12.append(temp_row)
##########
J13 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#tau8
              for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        J13.append(temp_row)
##########
J14 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#zeta3
          for j in range(1, n+d+1):
             w = 0
             if i==l and k==j:
                 w=1
             temp_row.append(w)
        J14.append(temp_row)
##########
J15 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta4 nodes
          for j in range(1, n+d+1):
                w = 0
                if i==l and k==j:
                    w=1
                temp_row.append(w)
        J15.append(temp_row)
##########
for i in range(len(J11)):
    concatenated_row = J11[i] + J12[i] +J13[i] + J14[i]+J15[i] 
    W11.append(concatenated_row)    
######################
## tau9 nodes as identity map 
J16 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             temp_row.append(w)
        J16.append(temp_row)
##########
J17 = []
for i in range(1, n+d+1):
        temp_row = []
        for j in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        J17.append(temp_row)
##########
J18 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#tau8
              for j in range(1, d+1):
                w = 0
                if i==l:
                    w=1
                temp_row.append(w)
        J18.append(temp_row)
##########
J19 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#zeta3
          for j in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        J19.append(temp_row)
##########
J20 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# zeta4 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J20.append(temp_row)
##########
for i in range(len(J16)):
    concatenated_row = J16[i] + J17[i] +J18[i] + J19[i]+J20[i] 
    W11.append(concatenated_row) 
######################
# print("weight matrix for twelfth layer/eleventh hidden layer")
# print(W11)
#####################

# #Bias matrix for twelfth layer/eleventh hidden layer

#B11=[] is zero matrix 
################################## 
L11 = []  # nu1, tau9
for i in range(len(W11)):
    temp_row = []
    L11_i_entry = np.maximum((np.dot(W11[i], L10)), 0)
    L11.append(L11_i_entry)
#################################
# print('Printing nu1, tau9 nodes for twelfth layer/eleventh hidden layer')
# for i in L11:
#     print(i)
############
# To construct weight matrix for thirteenth layer/twelfth hidden layer is L12=W12*L11+B12
W12 = []
# tau10 nodes  
J21 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             if i==l:
                 w=C
             temp_row.append(w)
        J21.append(temp_row)
##########
J22 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        J22.append(temp_row)
##########
J23 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#nu1
              for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J23.append(temp_row)
##########
J24 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#tau9
             w = 0
             temp_row.append(w)
        J24.append(temp_row)
##########
for i in range(len(J21)):
    concatenated_row = J21[i] + J22[i] +J23[i] + J24[i]
    W12.append(concatenated_row)    
######################
## tau11 nodes as identity map 
J26 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             temp_row.append(w)
        J26.append(temp_row)
##########
J27 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## eta23
             w = 0
             if i==k:
                 w=C
             temp_row.append(w)
        J27.append(temp_row)
##########
J28 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#nu1
              for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J28.append(temp_row)
##########
J29 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#tau9
             w = 0
             if i==k:
                 w=1
             temp_row.append(w)
        J29.append(temp_row)
##########
for i in range(len(J26)):
    concatenated_row = J26[i] + J27[i] +J28[i] + J29[i]
    W12.append(concatenated_row)    
######################
## nu1 nodes as identity map 
J31 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# eta22
             w = 0
             temp_row.append(w)
        J31.append(temp_row)
##########
J32 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for j in range(1, n+d+1):## eta23
             w = 0
             temp_row.append(w)
        J32.append(temp_row)
##########
J33 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#nu1
            for j in range(1, n+d+1):
                w = 0
                if i==l and k==j:
                    w=1
                temp_row.append(w)
        J33.append(temp_row)
##########
J34 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#tau9
             w = 0
             temp_row.append(w)
        J34.append(temp_row)
##########
for i in range(len(J31)):
    concatenated_row = J31[i] + J32[i] +J33[i] + J34[i]
    W12.append(concatenated_row)    
######################
# print("weight matrix for thirteenth layer/twelfth hidden layer")
# print(W12)
#####################

# #Bias matrix for thirteenth layer/twelfth hidden layer

B12=[] 
# bias matrix for tau10 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = U[i-1]-C
          temp_row.append(b)
      B12.append(temp_row)
# bias matrix for tau11 nodes

for i in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = -C
          temp_row.append(b)
      B12.append(temp_row)
        
# bias matrix for nu1 nodes 

for i in range(1, n+d+1):
      for j in range(1, n+d+1):
        temp_row = []
        for l in range(1):
            b=0
            temp_row.append(b)
        B12.append(temp_row)           
# print('Printing B12')
# for i in B12:
#     print(i)      
################################## 
################################## 
L12 = []  # tau10, tau11
for i in range(len(W12)):
    temp_row = []
    L12_i_entry = np.maximum((np.dot(W12[i], L11)+B12[i]), 0)
    L12.append(L12_i_entry)
#################################
# print('Printing tau10, tau11 nodes for thirteenth layer/twelfth hidden layer')
# for i in L12:
#     print(i)
############
# To construct weight matrix for fourteenth layer/thirteenth hidden layer is L13=W13*L12+B13
W13 = []
# u' nodes  
J35 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau10
             w = 0
             if i==l:
                 w=1
             temp_row.append(w)
        J35.append(temp_row)
##########
J36 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):## tau11
             w = 0
             if i==k:
                 w=1
             temp_row.append(w)
        J36.append(temp_row)
##########
J37 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#nu1
              for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J37.append(temp_row)
##########
for i in range(len(J35)):
    concatenated_row = J35[i] + J36[i] +J37[i]
    W13.append(concatenated_row)       
######################
## nu1(v') nodes as identity map 
J38 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau10
             w = 0
             temp_row.append(w)
        J38.append(temp_row)
##########
J39 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for j in range(1, n+d+1):## tau11
             w = 0
             temp_row.append(w)
        J39.append(temp_row)
##########
J40 = []
for i in range(1, n+d+1):
    for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#nu1
            for j in range(1, n+d+1):
                w = 0
                if i==l and k==j:
                    w=1
                temp_row.append(w)
        J40.append(temp_row)
##########
for i in range(len(J38)):
    concatenated_row = J38[i] + J39[i] +J40[i] 
    W13.append(concatenated_row)    
######################
######################
# print("weight matrix for fourteenth layer/thirteenth hidden layer")
# print(W13)
#####################

# #Bias matrix for output /fourteenth layer/thirteenth layer

#B13=[] is zero matrix     
################################## 
Y = []  # 
for i in range(len(W13)):
    temp_row = []
    Y_i_entry = np.maximum((np.dot(W13[i], L12)), 0)
    Y.append(Y_i_entry)
#################################
# print('Printing Output matrix')
# for i in Y:
#     print(i)
############
nd = n + d  # 7

# Flatten Y
flat_Y = [int(x[0]) if x[0].is_integer() else x[0] for x in Y]

# First nd values are Uprime
Uprime = flat_Y[:nd]

# The remaining values for Vprime
adj_values = flat_Y[nd: nd + nd*nd]  # take next nd*nd values only

# Build Vprime row by row
Vprime = [adj_values[i*nd:(i+1)*nd] for i in range(nd)]

# print("Uprime =", Uprime)
# print("Vprime = [")
# for row in Vprime:
#     print("    ", row, ",")
# print("]")
#########################################
# Find indices of entries that are NOT B
indices_to_keep = [i for i, val in enumerate(Uprime) if val != B]

# Remove 1000s from Uprime
Lprime = [Uprime[i] for i in indices_to_keep]

# Remove rows and columns in Vprime corresponding to 1000s in Uprime
Aprime = [[Vprime[i][j] for j in indices_to_keep] for i in indices_to_keep]

print("L' that is label matrix of output graph is=", Lprime)
print("A' that is adjacency matrix of output graph is = [")
for row in Aprime:
    print("   ", row, ",")
print("]")