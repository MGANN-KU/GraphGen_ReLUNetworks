# -*- coding: utf-8 -*-
"""
Created on Fri July 25,2025 at 10:28:00

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
# one- dimentional label matrix
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
x = [5, 3, 3, 5, 2, 3]
X = x  
print("Input:")
print("d:", d)
print("m:", m)
print("x:", X)
print("L:", L)
print("A:")
for row in A:
    print(row)
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
U_prime = pad_adjacency_matrix(A, d, B)

# Print the padded matrix
# print("Padded Adjacency Matrix U prime:")
# for row in U_prime:
#     print(row)
#### Flatten 2D list to 1D
tau1 = [item for row in U_prime for item in row]
# print(tau1)
#####################################################################
# To construct weight matrix for second layer/first hidden layer is L1=W1*X+B1
W1 = []

## (eta1, eta2 to solve delta(x_j, x_{j+d}))
for j in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
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
      for k in range(1, 2*d+1):
          w = 0
          if j == k:
                w = -1/eps
          if j+d == k:
                  w=1/eps
          temp_row.append(w)
      W1.append(temp_row)
####
## (alpha1, alpha2 to solve delta(x_j, i)
for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j == k:
                w = 1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## (beta1, beta2 to solve delta(x_j, i)
for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j == k:
                w = -1/eps
          temp_row.append(w)
      W1.append(temp_row)
####
## alpha3, alpha4 to solve delta(x_{j+d}, l)
for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j+d == k:
                w = 1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## beta3, beta4 to solve delta(x_{j+d}, l)
for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j+d == k:
                w = -1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## (alpha5, alpha6 to solve delta(x_j, l)
for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j == k:
                w = 1/eps
          temp_row.append(w)
      W1.append(temp_row)
####
## (beta5, beta6 to solve delta(x_j, l)
for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j == k:
                w = -1/eps
          temp_row.append(w)
      W1.append(temp_row)
####
## alpha7, alpha8 to solve delta(x_{j+d},i)
for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j+d == k:
                w = 1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## beta7, beta8 to solve delta(x_{j+d}, i)
for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2): 
      temp_row = []
      for k in range(1, 2*d+1):
          w = 0
          if j+d == k:
                w = -1/eps
          temp_row.append(w)
      W1.append(temp_row)
#####
## copy x_j
for j in range(1, 2*d+1):  
      temp_row = []
      for k in range(1, 2*d+1):
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
      
# bias matrix for alpha1, alpha2 to solve delta(x_j, i)

for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0:
              b=(-i/eps)+1
          else:
              b=(-i/eps)
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta1, beta2 to solve delta(x_j, i)

for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0:
              b=(i/eps)+1
          else:
              b=(i/eps)
          temp_row.append(b)
      B1.append(temp_row)
############
# bias matrix for alpha3, alpha4 to solve delta(x_{j+d}, l)

for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=(-l/eps)+1
          else:
              b=(-l/eps)
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta3, beta4 to solve delta(x_{j+d}, l)

for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=(l/eps)+1
          else:
              b=(l/eps)
          temp_row.append(b)
      B1.append(temp_row)
############

# bias matrix for alpha5, alpha6 to solve delta(x_j, l)

for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=(-l/eps)+1
          else:
              b=(-l/eps)
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta5, beta6 to solve delta(x_j, l)

for l in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=(l/eps)+1
          else:
              b=(l/eps)
          temp_row.append(b)
      B1.append(temp_row)
############
# bias matrix for alpha7, alpha8 to solve delta(x_{j+d}, i)

for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0:
              b=(-i/eps)+1
          else:
              b=(-i/eps)
          temp_row.append(b)
      B1.append(temp_row)

# bias matrix for beta7, beta8 to solve delta(x_{j+d}, i)

for i in range(1, n+d+1):
  for j in range(1, d+1):  
    for q in range(2):
      temp_row = []
      for l in range(1):
          b = 0
          if q==0:
              b=(i/eps)+1
          else:
              b=(i/eps)
          temp_row.append(b)
      B1.append(temp_row)
      
## bias matrix for x_j
for j in range(1, 2*d+1):  
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B1.append(temp_row)      
############
# # for i in B1:
# #     print(i)
# ##################################
L1 = []  #eta1, eta2, eta3, eta4, alpha1, alpha2, beta1, beta2,alpha3, alpha4, beta3, beta4,
#alpha5, alpha6, beta5, beta6, alpha7, alpha8, beta7, beta8
for i in range(len(W1)):
    temp_row = []
    L1_i_entry = np.maximum((np.dot(W1[i], X)+B1[i]), 0)
    L1.append(L1_i_entry)
# ############
# print('Printing eta1, eta2, eta3, eta4, alpha1, alpha2, beta1, beta2,alpha3, alpha4, beta3, beta4,alpha5, alpha6, beta5, beta6, alpha7, alpha8, beta7, beta8 nodes of second layer/first hidden layer')
# for i in L1:
#       print(i)
############
# To construct weight matrix for third layer/second hidden layer is L2=W2*L1+B2
W2 = []
# for eta5 that corresponding to b^1_il
A1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =-1
                else:
                    w=1
            temp_row.append(w)
        A1.append(temp_row)
#####
A2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w = -1
                else:
                    w= 1
            temp_row.append(w)
        A2.append(temp_row)
##########
A3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A3.append(temp_row)
##########
A4 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A4.append(temp_row)
##########
A5 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A5.append(temp_row)
##########
A6 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A6.append(temp_row)
##########
A7 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha5, alpha6 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A7.append(temp_row)
##########
A8 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta5, beta6 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A8.append(temp_row)
##########
A9 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha7, alpha8 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A9.append(temp_row)
##########
A10 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta7, beta8 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A10.append(temp_row)
##########
A11 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, 2*d+1): 
            w = 0
            temp_row.append(w)
        A11.append(temp_row)
##########
for i in range(len(A1)):
    concatenated_row = A1[i] + A2[i]+ A3[i] + A4[i]+A5[i] + A6[i] + A7[i] + A8[i]+A9[i] + A10[i]+ A11[i]
    W2.append(concatenated_row)
# print("weight matrix for third layer/second hidden layer")
# print(W2)
#######################
# for eta'5 that corresponding to b^2_il
A12 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =-1
                else:
                    w=1
            temp_row.append(w)
        A12.append(temp_row)
#####
A13 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w = -1
                else:
                    w= 1
            temp_row.append(w)
        A13.append(temp_row)
##########
A14 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A14.append(temp_row)
##########
A15 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A15.append(temp_row)
##########
A16 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A16.append(temp_row)
##########
A17 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            temp_row.append(w)
        A17.append(temp_row)
##########
A18 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha5, alpha6 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A18.append(temp_row)
##########
A19 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta5, beta6 nodes
          for k in range(1, d+1):  
           for q in range(2):
            w = 0
            if l==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        A19.append(temp_row)
##########
A20 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha7, alpha8 nodes
          for k in range(1, d+1):  
            for q in range(2):
             w = 0
             if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
             temp_row.append(w)
        A20.append(temp_row)
##########
A21 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta7, beta8 nodes
          for k in range(1, d+1):  
            for q in range(2):
             w = 0
             if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
             temp_row.append(w)
        A21.append(temp_row)
##########
A22 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, 2*d+1): 
             w = 0
             temp_row.append(w)
        A22.append(temp_row)
##########
for i in range(len(A12)):
    concatenated_row = A12[i]+A13[i]+ A14[i]+A15[i]+A16[i]+A17[i]+A18[i]+A19[i]+A20[i]+A21[i]+A22[i]
    W2.append(concatenated_row)
# #######################
# delta(x_j, i) for 1<=i<=n
A23 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        A23.append(temp_row)
#####
A24 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        A24.append(temp_row)
##########
A25 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
            for q in range(2):
                w = 0
                if i==p and j == k:
                  if q==0:
                     w = 1
                  else:
                    w= -1
                temp_row.append(w)
        A25.append(temp_row)
##########
A26 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
            for q in range(2):
             w = 0
             if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
             temp_row.append(w)
        A26.append(temp_row)
##########
A27 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A27.append(temp_row)
##########
A28 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A28.append(temp_row)
##########
A29 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha5, alpha6 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A29.append(temp_row)
##########
A30 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta5, beta6 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A30.append(temp_row)
##########
A31 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha7, alpha8 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A31.append(temp_row)
##########
A32 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta7, beta8 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        A32.append(temp_row)
##########
A44 = []
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, 2*d+1): 
              w = 0
              temp_row.append(w)
        A44.append(temp_row)
##########
for i in range(len(A23)):
    concatenated_row = A23[i] + A24[i]+ A25[i] + A26[i]+A27[i] + A28[i]+ A29[i] + A30[i]+A31[i] + A32[i]+ A44[i]
    W2.append(concatenated_row)
#####################
# #############################
## copy x_j
M1 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        M1.append(temp_row)
#####
M2 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        M2.append(temp_row)
##########
M3 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha1, alpha2 nodes
          for k in range(1, d+1):  
            for q in range(2):
               w = 0
               temp_row.append(w)
        M3.append(temp_row)
##########
M4 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta1, beta2 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M4.append(temp_row)
##########
M5 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M5.append(temp_row)
##########
M6 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M6.append(temp_row)
##########
M7 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha5, alpha6 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M7.append(temp_row)
##########
M8 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta5, beta6 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M8.append(temp_row)
##########
M9 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha7, alpha8 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M9.append(temp_row)
##########
M10 = []
for j in range(1, 2*d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta7, beta8 nodes
          for k in range(1, d+1):  
            for q in range(2):
              w = 0
              temp_row.append(w)
        M10.append(temp_row)
##########
M11 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, 2*d+1):
              w = 0
              if j==k:
                  w=1
              temp_row.append(w)
        M11.append(temp_row)
##########
for i in range(len(M1)):
    concatenated_row = M1[i] + M2[i]+ M3[i] + M4[i]+M5[i] + M6[i] + M7[i] + M8[i]+M9[i] + M10[i] + M11[i]
    W2.append(concatenated_row)
# # print("weight matrix for third layer/second hidden layer")
# # print(W2)
# #############################
# Bias matrix for third layer/second hidden layer

B2 = []
# bias matrix for eta5 that corresponding to b^1_il
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -2
          temp_row.append(b)
        B2.append(temp_row)
# bias matrix for eta'5 that corresponding to b^2_il
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -2
          temp_row.append(b)
        B2.append(temp_row)       

# bias matrix for delta(x_j, i)
for i in range(1, n+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -1
          temp_row.append(b)
        B2.append(temp_row)  

# bias for x_j
for j in range(1, 2*d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B2.append(temp_row)      
#################################
L2 = []  # eta5, eta'5, delta(x_j,i), eta9 nodes
for i in range(len(W2)):
    temp_row = []
    L2_i_entry = np.maximum((np.dot(W2[i], L1)+B2[i]), 0)
    L2.append(L2_i_entry)
##################################
# print('Printing eta5, eta prime 5, delta(x_j,i), d prime nodes for third layer/second hidden layer')
# for i in L2:
#     print(i)
###################
# To construct weight matrix for fourth layer/third hidden layer is L3=W3*L2+B3
W3 = []
# gamma1 nodes for t_ik
A45 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^1_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            if i == p and l==k:
              w = 1
            temp_row.append(w)
    A45.append(temp_row)
# ##########
A46 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^2_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            if i == p and l==k:
              w = 1
            temp_row.append(w)
    A46.append(temp_row)
# ##########
A47 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    temp_row = []
    for p in range(1, n+1):# delta(x_j,i)
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A47.append(temp_row)
##########
A49 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    temp_row = []
    for p in range(1, 2*d+1):
            w = 0
            temp_row.append(w)
    A49.append(temp_row)
##########
for i in range(len(A45)):
    concatenated_row = A45[i] +A46[i] + A47[i]+ A49[i]
    W3.append(concatenated_row)   
# #######################
# delta(x_j,i) as identity map
A50 = []
for i in range(1, n+1):
  for l in range(1, d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^1_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A50.append(temp_row)
# ##########
A51 = []
for i in range(1, n+1):
  for l in range(1, d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^2_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A51.append(temp_row)
# ##########
A52 = []
for i in range(1, n+1):
  for l in range(1, d+1):
    temp_row = []
    for p in range(1, n+1):# delta(x_j,i)
        for j in range(1, d+1):
            w = 0
            if i==p and l==j:
                w=1
            temp_row.append(w)
    A52.append(temp_row)
##########
##########
A54 = []
for i in range(1, n+1):
  for l in range(1, d+1):
    temp_row = []
    for p in range(1, 2*d+1):
            w = 0
            temp_row.append(w)
    A54.append(temp_row)
##########
for i in range(len(A50)):
    concatenated_row = A50[i] +A51[i] + A52[i]+ A54[i]
    W3.append(concatenated_row)   
##########################    
#x_j as identity map
A55 = []
for i in range(1, 2*d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^1_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A55.append(temp_row)
# ##########
A56 = []
for i in range(1, 2*d+1):
    temp_row = []
    for p in range(1, n+d+1):# b^2_il
      for k in range(1, n+d+1):
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A56.append(temp_row)
##########
A57 = []
for i in range(1, 2*d+1):
    temp_row = []
    for p in range(1, n+1):# 
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
    A57.append(temp_row)
##########
A59 = []
for j in range(1, 2*d+1):
    temp_row = []
    for k in range(1, 2*d+1):
            w = 0
            if j==k:
                w=1
            temp_row.append(w)
    A59.append(temp_row)
##########
for i in range(len(A55)):
    concatenated_row = A55[i] +A56[i] + A57[i] + A59[i] 
    W3.append(concatenated_row)
###############
#######################
# Bias matrix for fourth layer/third hidden layer

#B3 = [] is zero matrix        
##################################
L3 = []  # gamma1 nodes
for i in range(len(W3)):
    temp_row = []
    L3_i_entry = np.maximum((np.dot(W3[i], L2)), 0)
    L3.append(L3_i_entry)
##################################
# print('Printing gamma1, gamma2 , gamma3, gamma4 nodes for fourth layer/third hidden layer')
# for i in L3:
#     print(i)
###################
###########################
# To construct weight matrix for fifth layer/fouth hidden layer is L4=W4*L3+B4
W4 = []
# tau2 nodes(correspond to t'_il)i.e, the matrix after removing the indicated edges
D1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                if i == j and l==k:
                    w = -1
                temp_row.append(w)
            D1.append(temp_row)
##########
D2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+1):
              for k in range(1, d+1):
                w = 0
                temp_row.append(w)
            D2.append(temp_row)
##########
# ##########
D3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, 2*d+1):# x_j node
                w = 0
                temp_row.append(w)
            D3.append(temp_row)
##########
##########
for i in range(len(D1)):
    concatenated_row = D1[i]+ D2[i] +D3[i]  
    W4.append(concatenated_row)
#######################
# delta(x_j,i) as identity map
D5 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D5.append(temp_row)
##########
D6 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for j in range(1, n+1):
              for k in range(1, d+1):
                w = 0
                if i==j and l==k:
                    w=1
                temp_row.append(w)
            D6.append(temp_row)
###########
D7 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for j in range(1, 2*d+1):# x_j node
                w = 0
                temp_row.append(w)
            D7.append(temp_row)
##########
for i in range(len(D5)):
    concatenated_row = D5[i]+ D6[i] +D7[i] 
    W4.append(concatenated_row)
#######################
# x_j as identity map
D9 = []
for i in range(1, 2*d+1):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D9.append(temp_row)
###########
D10 = []
for i in range(1, 2*d+1):
            temp_row = []
            for j in range(1, n+1):
              for k in range(1, d+1):
                w = 0
                temp_row.append(w)
            D10.append(temp_row)
# ##########
D11 = []
for i in range(1, 2*d+1):
            temp_row = []
            for j in range(1, 2*d+1):# x_j node
                w = 0
                if i==j:
                    w=1
                temp_row.append(w)
            D11.append(temp_row)
##########
for i in range(len(D9)):
    concatenated_row = D9[i]+ D10[i] +D11[i]  
    W4.append(concatenated_row)
#######################
#####################
# Bias matrix for fifth layer/fourth hidden layer

B4 = []

# bias matrix for tau2 nodes(correspond to t'_il)i.e, the matrix after removing the indicated edges

for i in range(1, n+d+1):
    for l in range(1, n+d+1):
        temp_row = []
        for k in range(1):
          b = tau1[(i-1) * (n+d) + (l-1)]  # Access correct element from flattened matrix
          temp_row.append(b)
        B4.append(temp_row)
# bias matrix for delta

for i in range(1, n+1):
    for l in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = 0 
          temp_row.append(b)
        B4.append(temp_row)
        
# bias for x_{j}
for i in range(1, 2*d+1):
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
L4 = []  # tau2 nodes
for i in range(len(W4)):
    temp_row = []
    L4_i_entry = np.maximum((np.dot(W4[i], L3)+B4[i]), 0)
    L4.append(L4_i_entry)
    # print('this is index i:', i)
    # print('this is the value L4[i]:', L4_i_entry)
###################################
# print('Printing tau2 nodes for fifth layer/fouth hidden hidden layer')
# for i in L4:
#     print(i)
##################
# To construct weight matrix for sixth layer/fifth hidden layer is L5=W5*L4+B5
W5 = []
# tau2 nodes as identity map
E1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                if i==k and l==j:
                    w=1
                temp_row.append(w)
        E1.append(temp_row)
##########
E2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):
          for j in range(1, d+1):
                w = 0
                temp_row.append(w)
        E2.append(temp_row)
##########
E3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, 2*d+1):
                w = 0
                temp_row.append(w)
        E3.append(temp_row)
##########
for i in range(len(E1)):
    concatenated_row = E1[i] + E2[i] + E3[i]
    W5.append(concatenated_row)
#######################
# t'' to check the degree of nodes are zero or not
E4 = []
for i in range(1, n+1):
        temp_row = []
        for l in range(1, n+d+1):# tau2 nodes
          for k in range(1, n+d+1):
                w = 0
                if i==l and k<=n:
                    w=1
                temp_row.append(w)
        E4.append(temp_row)
##########
E5 = []
for i in range(1, n+1):
        temp_row = []
        for l in range(1, n+1):
          for k in range(1, d+1):
                w = 0
                temp_row.append(w)
        E5.append(temp_row)
##########
E6 = []
for i in range(1, n+1):
        temp_row = []
        for k in range(1, 2*d+1):
                w = 0
                temp_row.append(w)
        E6.append(temp_row)
##########
for i in range(len(E4)):
    concatenated_row = E4[i] + E5[i] + E6[i]
    W5.append(concatenated_row)
#######################
# delta(x_j,i) as identity map
E7 = []
for i in range(1, n+1):
  for l in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E7.append(temp_row)
##########
E8 = []
for i in range(1, n+1):
  for l in range(1, d+1):
        temp_row = []
        for k in range(1, n+1):
          for j in range(1, d+1):
                w = 0
                if i==k and l==j:
                    w=1
                temp_row.append(w)
        E8.append(temp_row)
##########
E9 = []
for i in range(1, n+1):
  for l in range(1, d+1):
        temp_row = []
        for k in range(1, 2*d+1):
                w = 0
                temp_row.append(w)
        E9.append(temp_row)
##########
##########
for i in range(len(E7)):
    concatenated_row = E7[i] + E8[i] + E9[i]
    W5.append(concatenated_row)
#######################
# x_j as identity map
E10 = []
for i in range(1, 2*d+1):
        temp_row = []
        for l in range(1, n+d+1):# tau2 nodes
          for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E10.append(temp_row)
##########
E11 = []
for i in range(1, 2*d+1):
        temp_row = []
        for l in range(1, n+1):
          for k in range(1, d+1):
                w = 0
                temp_row.append(w)
        E11.append(temp_row)
##########
E12 = []
for i in range(1, 2*d+1):
        temp_row = []
        for k in range(1, 2*d+1):
                w = 0
                if i==k:
                    w=1
                temp_row.append(w)
        E12.append(temp_row)
##########
for i in range(len(E10)):
    concatenated_row = E10[i] + E11[i]+ E12[i]
    W5.append(concatenated_row)
#######################
# #Bias matrix for sixth layer/fifth hidden layer
#B5 = [] is a zero matrix
##################################
L5 = []  # t'' nodes to check whether the degree of the node is zero or not
for i in range(len(W5)):
    temp_row = []
    L5_i_entry = np.maximum((np.dot(W5[i], L4)), 0)
    L5.append(L5_i_entry)
###################################
# print('Printing t double prime nodes for sixth layer/fifth hidden layer')
# for i in L5:
#     print(i)
###################
# To construct weight matrix for seventh layer/sixth hidden layer is L6=W6*L5+B6
W6 = []
# tau2 nodes as identity map
F1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):
              for j in range(1, n+d+1):# tau2 nodes
                  w = 0
                  if k == i and l==j:
                      w = 1
                  temp_row.append(w)
            F1.append(temp_row)
##########
F2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+1):# t'' nodes
                  w = 0
                  temp_row.append(w)
            F2.append(temp_row)

##########
F3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+1):# delta nodes
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
            F3.append(temp_row)

##########
F4 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, 2*d+1):#xj nodes
                  w = 0
                  temp_row.append(w)
            F4.append(temp_row)
##########
for i in range(len(F1)):
    concatenated_row = F1[i] + F2[i] + F3[i] + F4[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# alpha'1, alpha'2 nodes to calculate delta(t''_i,0)
F5 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for j in range(1, n+d+1):# tau2 nodes
                  w = 0
                  temp_row.append(w)
            F5.append(temp_row)
##########
F6 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, n+1):# t'' nodes
                  w = 0
                  if i==k:
                      w=1/eps
                  temp_row.append(w)
            F6.append(temp_row)

##########
F7 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for j in range(1, n+1):# delta nodes
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
            F7.append(temp_row)

##########
F8 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, 2*d+1):#xj nodes
                  w = 0
                  temp_row.append(w)
            F8.append(temp_row)
##########
for i in range(len(F5)):
    concatenated_row = F5[i] + F6[i] + F7[i] + F8[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# beta'1, beta'2 nodes to calculate delta(t''_i,0)
F9 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for j in range(1, n+d+1):# tau2 nodes
                  w = 0
                  temp_row.append(w)
            F9.append(temp_row)
##########
F10 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, n+1):# t'' nodes
                  w = 0
                  if i==k:
                      w=-1/eps
                  temp_row.append(w)
            F10.append(temp_row)

##########
F11 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for j in range(1, n+1):# delta nodes
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
            F11.append(temp_row)

##########
F12 = []
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1, 2*d+1):#xj nodes
                  w = 0
                  temp_row.append(w)
            F12.append(temp_row)
##########
for i in range(len(F9)):
    concatenated_row = F9[i] + F10[i] + F11[i] + F12[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# delta(xj,i) nodes 
F13 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for k in range(1, n+d+1):
              for j in range(1, n+d+1):# tau2 nodes
                  w = 0
                  temp_row.append(w)
            F13.append(temp_row)
##########
F14 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for k in range(1, n+1):# t'' nodes
                  w = 0
                  temp_row.append(w)
            F14.append(temp_row)

##########
F15 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for j in range(1, n+1):# delta nodes
              for k in range(1, d+1):
                  w = 0
                  if j == i and l==k:
                      w = 1
                  temp_row.append(w)
            F15.append(temp_row)

##########
F16 = []
for i in range(1, n+1):
  for l in range(1, d+1):
            temp_row = []
            for k in range(1, 2*d+1):#xj nodes
                  w = 0
                  temp_row.append(w)
            F16.append(temp_row)
##########
for i in range(len(F13)):
    concatenated_row = F13[i] + F14[i] + F15[i] + F16[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# xj nodes as identity map 
F17 = []
for i in range(1, 2*d+1):
            temp_row = []
            for k in range(1, n+d+1):
              for j in range(1, n+d+1):# tau2 nodes
                  w = 0
                  temp_row.append(w)
            F17.append(temp_row)
##########
F18 = []
for i in range(1, 2*d+1):
            temp_row = []
            for k in range(1, n+1):# t'' nodes
                  w = 0
                  temp_row.append(w)
            F18.append(temp_row)

##########
F19 = []
for i in range(1, 2*d+1):
            temp_row = []
            for j in range(1, n+1):# delta nodes
              for k in range(1, d+1):
                  w = 0
                  temp_row.append(w)
            F19.append(temp_row)

##########
F20 = []
for i in range(1, 2*d+1):
            temp_row = []
            for k in range(1, 2*d+1):#xj nodes
                  w = 0
                  if i==k:
                      w=1
                  temp_row.append(w)
            F20.append(temp_row)
##########
for i in range(len(F17)):
    concatenated_row = F17[i] + F18[i] + F19[i] + F20[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# #######################
# print("weight matrix for seventh layer/sixth hidden layer")
# print(W6)
#####################
# #Bias matrix for seventh layer/sixth hidden layer
B6 = []

# bias matrix tau2 nodes as identity map
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B6.append(temp_row)
# bias matrix for alpha'1, alpha'2 nodes
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b=1
                temp_row.append(b)
            B6.append(temp_row)

# bias matrix for beta'1, beta'2 nodes
for i in range(1, n+1):
  for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b=1
                temp_row.append(b)
            B6.append(temp_row)

# bias for delta(xj,i) nodes 

for i in range(1, n+1):
  for l in range(1, d+1): 
     temp_row = []
     for k in range(1):
         b = 0
         temp_row.append(b)
     B6.append(temp_row)    

# bias for xj nodes 

for i in range(1, 2*d+1): 
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
L6 = []  # alpha'1, alpha'2, beta'1, beta'2, delta(t'',0) nodes
for i in range(len(W6)):
    temp_row = []
    L6_i_entry = np.maximum((np.dot(W6[i], L5)+B6[i]), 0)
    L6.append(L6_i_entry)
##################################
# print('Printing alpha prime1, alpha prime2, beta prime1, beta prime2, delta(t'',0) nodes for seventh layer/sixth hidden layer')
# for i in L6:
#     print(i)
############
# To construct weight matrix for eighth layer/seventh hidden layer is L7=W7*L6+B7
W7 = []
# tau2 nodes as identity map
G1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                if i == k and j==l:
                        w = 1
                temp_row.append(w)
        G1.append(temp_row)

##########
G2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):# alpha'1 alpha'2 nodes
           for q in range(2):
                    w = 0
                    temp_row.append(w)
        G2.append(temp_row)
##########
G3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):# beta'1, beta'2 nodes
           for q in range(2):
                    w = 0
                    temp_row.append(w)
        G3.append(temp_row)
##########
G4 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):
          for j in range(1, d+1):
                    w = 0
                    temp_row.append(w)
        G4.append(temp_row)
##########
G5 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for p in range(1, 2*d+1):
                    w = 0
                    temp_row.append(w)
        G5.append(temp_row)
##########
for i in range(len(G1)):
    concatenated_row = G1[i] + G2[i] + G3[i] + G4[i] + G5[i] 
    W7.append(concatenated_row)
# #######################
# tau3 nodes corresponding to delta(xj,i)+delta(t''_i,0)-1
G6 = []
for i in range(1, n+1):
  for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        G6.append(temp_row)

##########
G7 = []
for i in range(1, n+1):
  for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+1):# alpha'1 alpha'2 nodes
           for q in range(2):
                    w = 0
                    if i == k:
                        if q==0:
                            w = 1
                        else:
                             w=-1
                    temp_row.append(w)
        G7.append(temp_row)
##########
G8 = []
for i in range(1, n+1):
  for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+1):# beta'1, beta'2 nodes
           for q in range(2):
                    w = 0
                    if i == k:
                        if q==0:
                            w = 1
                        else:
                             w=-1
                    temp_row.append(w)
        G8.append(temp_row)
##########
G9 = []
for i in range(1, n+1):
  for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+1):
          for l in range(1, d+1):
                    w = 0
                    if i==k and j==l:
                        w=1
                    temp_row.append(w)
        G9.append(temp_row)
##########
G10 = []
for i in range(1, n+1):
  for j in range(1, d+1):
        temp_row = []
        for p in range(1, 2*d+1):
                    w = 0
                    temp_row.append(w)
        G10.append(temp_row)
##########
for i in range(len(G6)):
    concatenated_row = G6[i] + G7[i] + G8[i] + G9[i] + G10[i] 
    W7.append(concatenated_row)
########################
# x_j as identity map
G11 = []
for i in range(1, 2*d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for l in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        G11.append(temp_row)

##########
G12 = []
for i in range(1, 2*d+1):
        temp_row = []
        for k in range(1, n+1):# alpha'1 alpha'2 nodes
           for q in range(2):
                    w = 0
                    temp_row.append(w)
        G12.append(temp_row)
##########
G13 = []
for i in range(1, 2*d+1):
        temp_row = []
        for k in range(1, n+1):# beta'1, beta'2 nodes
           for q in range(2):
                    w = 0
                    temp_row.append(w)
        G13.append(temp_row)
##########
G14 = []
for i in range(1, 2*d+1):
        temp_row = []
        for k in range(1, n+1):
          for l in range(1, d+1):
                    w = 0
                    temp_row.append(w)
        G14.append(temp_row)
##########
G15 = []
for i in range(1, 2*d+1):
        temp_row = []
        for j in range(1, 2*d+1):
                    w = 0
                    if i==j:
                        w=1
                    temp_row.append(w)
        G15.append(temp_row)
##########
for i in range(len(G11)):
    concatenated_row = G11[i] + G12[i] + G13[i] + G14[i] + G15[i] 
    W7.append(concatenated_row)
# #######################
# print("weight matrix for eighth layer/seventh hidden layer")
# print(W7)
#####################
# #Bias matrix for eighth layer/seventh hidden layer

B7 = []

# bias matrix for tau2

for i in range(1, n+d+1):
  for l in range(1, n+d+1): 
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B7.append(temp_row)
        
# bias matrix for tau3

for i in range(1, n+1):
  for l in range(1, d+1): 
      temp_row = []
      for k in range(1):
          b = -2
          temp_row.append(b)
      B7.append(temp_row)

# bias matrix for xj

for i in range(1, 2*d+1):
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B7.append(temp_row)              
# print('Printing B7')
# for i in B7:
#     print(i)
##################################
L7 = []  # tau3 nodes
for i in range(len(W7)):
    temp_row = []
    L7_i_entry = np.maximum((np.dot(W7[i], L6)+B7[i]), 0)
    L7.append(L7_i_entry)
##################################
# print('Printing tau3 nodes for eighth layer/seventh hidden layer')
# for i in L7:
#     print(i)
#####################
# To construct weight matrix for ninth layer/eighth hidden layer is L7=W7*L6+B7
W8 = []
# tau2 nodes as identity map 
H1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
          temp_row = []
          for p in range(1, n+d+1):# tau2 nodes
            for r in range(1, n+d+1): 
                    w = 0
                    if i == p and l==r:
                        w = 1
                    temp_row.append(w)
          H1.append(temp_row)
##########
H2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
          temp_row = []          
          for i in range(1, n+1):# tau3 nodes
            for l in range(1, d+1):
                    w = 0
                    temp_row.append(w)
          H2.append(temp_row)
##########
H3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
          temp_row = []          
          for p in range(1, 2*d+1):# xj nodes
                    w = 0
                    temp_row.append(w)
          H3.append(temp_row)
##########
for i in range(len(H1)):
    concatenated_row = H1[i] + H2[i] + H3[i] 
    W8.append(concatenated_row)
# print(W8)
######################
# x' nodes
H4 = []
for i in range(1, d+1):
          temp_row = []
          for p in range(1, n+d+1):# tau2 nodes
            for r in range(1, n+d+1): 
                    w = 0
                    temp_row.append(w)
          H4.append(temp_row)
##########
H5 = []
for k in range(1, d+1):
          temp_row = []          
          for i in range(1, n+1):# tau3 nodes
            for j in range(1, d+1):
                    w = 0
                    if k == j:
                        w = C
                    temp_row.append(w)
          H5.append(temp_row)
##########
H6 = []
for i in range(1, d+1):
          temp_row = []          
          for k in range(1, 2*d+1):# xj nodes
                    w = 0
                    if i == k:
                        w = 1
                    temp_row.append(w)
          H6.append(temp_row)
##########
for i in range(len(H4)):
    concatenated_row = H4[i] + H5[i] + H6[i] 
    W8.append(concatenated_row)
# print(W8)
######################
# x_{j+d} nodes as identity map
H7 = []
for i in range(d+1, 2*d+1):
          temp_row = []
          for p in range(1, n+d+1):# tau2 nodes
            for r in range(1, n+d+1): 
                    w = 0
                    temp_row.append(w)
          H7.append(temp_row)
##########
H8 = []
for k in range(d+1, 2*d+1):
          temp_row = []          
          for i in range(1, n+1):# tau3 nodes
            for j in range(1, d+1):
                    w = 0
                    temp_row.append(w)
          H8.append(temp_row)
##########
H9 = []
for i in range(d+1, 2*d+1):
          temp_row = []          
          for k in range(1, 2*d+1):# xj nodes
                    w = 0
                    if i == k:
                        w = 1
                    temp_row.append(w)
          H9.append(temp_row)
##########
for i in range(len(H7)):
    concatenated_row = H7[i] + H8[i] + H9[i] 
    W8.append(concatenated_row)
# print(W8)
######################
# print("weight matrix for ninth layer/eighth hidden layer")
# print(W8)
#####################
# #Bias matrix for ninth layer/eighth hidden layer

B8 = []

# bias matrix for tau2

for i in range(1, n+d+1):
  for l in range(1, n+d+1): 
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B8.append(temp_row)
        
# bias matrix for x'

for i in range(1, d+1): 
      temp_row = []
      for k in range(1):
          b = -C
          temp_row.append(b)
      B8.append(temp_row)

# bias matrix for x_{j+d}

for i in range(d+1, 2*d+1):
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B8.append(temp_row)              
# print('Printing B8')
# for i in B8:
#     print(i)
##################################
L8 = []  # x' nodes
for i in range(len(W8)):
    temp_row = []
    L8_i_entry = np.maximum((np.dot(W8[i], L7)+B8[i]), 0)
    L8.append(L8_i_entry)
#############################
# print('Printing x prime nodes for ninth layer/eighth hidden layer')
# for i in L8:
#     print(i)
############################
# To construct weight matrix for tenth layer/ninth hidden layer is L9=W9*L8+B9
W9 = []
# tau2 nodes as identity map 
H10 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
             w = 0
             if i==k and l==j:
               w = 1
             temp_row.append(w)
        H10.append(temp_row)
##########
H11 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1):
              w = 0
              temp_row.append(w)
        H11.append(temp_row)
##########
H12 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for j in range(d+1, 2*d+1):
              w = 0
              temp_row.append(w)
        H12.append(temp_row)
##########
for i in range(len(H10)):
    concatenated_row = H10[i] + H11[i]+ H12[i] 
    W9.append(concatenated_row)    
#######################
# alpha3, alpha4 nodes to calculate delta(x'j,i) nodes 
H13 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H13.append(temp_row)
##########
H14 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1, d+1):
              w = 0
              if j==k:
                  w=1/eps
              temp_row.append(w)
        H14.append(temp_row)
##########
H15 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for p in range(d+1, 2*d+1):
              w = 0
              temp_row.append(w)
        H15.append(temp_row)
##########
for i in range(len(H13)):
    concatenated_row = H13[i] + H14[i]+ H15[i] 
    W9.append(concatenated_row)    
#######################
# beta3, beta4 nodes to calculate delta(x'j,i) nodes 
H16 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H16.append(temp_row)
##########
H17 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1, d+1):
              w = 0
              if j==k:
                  w=-1/eps
              temp_row.append(w)
        H17.append(temp_row)
##########
H18 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for p in range(d+1, 2*d+1):
              w = 0
              temp_row.append(w)
        H18.append(temp_row)
##########
for i in range(len(H16)):
    concatenated_row = H16[i] + H17[i]+ H18[i] 
    W9.append(concatenated_row)
####################### 
# alpha5, alpha6 nodes to calculate delta(x'j,k) nodes 
H19 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H19.append(temp_row)
##########
H20 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1, d+1):
              w = 0
              if j==i:
                  w=1/eps
              temp_row.append(w)
        H20.append(temp_row)
##########
H21 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for p in range(d+1, 2*d+1):
              w = 0
              temp_row.append(w)
        H21.append(temp_row)
##########
for i in range(len(H19)):
    concatenated_row = H19[i] + H20[i]+ H21[i] 
    W9.append(concatenated_row)    
#######################
# beta5, beta6 nodes to calculate delta(x'j,k) nodes 
H22 = []
for k in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H22.append(temp_row)
##########
H23 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1, d+1):
              w = 0
              if j==k:
                  w=-1/eps
              temp_row.append(w)
        H23.append(temp_row)
##########
H24 = []
for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for p in range(d+1, 2*d+1):
              w = 0
              temp_row.append(w)
        H24.append(temp_row)
##########
for i in range(len(H22)):
    concatenated_row = H22[i] + H23[i]+ H24[i] 
    W9.append(concatenated_row)
####################### 
# alpha7, alpha8 nodes to calculate delta(x'j,x_{j+d}) nodes 
H25 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H25.append(temp_row)
##########
H26 = []
for j in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for k in range(1, d+1):
          w = 0
          if j == k:
                w = 1/eps
          temp_row.append(w)
      H26.append(temp_row)
##########
H27 = []
for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(d+1, 2*d+1):
              w = 0
              if j+d == k:
                w=-1/eps
              temp_row.append(w)
        H27.append(temp_row)
##########
for i in range(len(H25)):
    concatenated_row = H25[i] + H26[i]+ H27[i] 
    W9.append(concatenated_row)
####################### 
# beta7, beta8 nodes to calculate delta(x'j,x_{j+d}) nodes 
H28 = []
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, n+d+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H28.append(temp_row)
##########
H29 = []
for j in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for k in range(1, d+1):
          w = 0
          if j == k:
                w = -1/eps
          temp_row.append(w)
      H29.append(temp_row)
##########
H30 = []
for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(d+1, 2*d+1):
              w = 0
              if j+d == k:
                w=1/eps
              temp_row.append(w)
        H30.append(temp_row)
##########
for i in range(len(H28)):
    concatenated_row = H28[i] + H29[i]+ H30[i] 
    W9.append(concatenated_row)
#######################
# print("weight matrix for tenth layer/ninth hidden layer")
# print(W9)
#####################
# #Bias matrix for tenth layer/ninth hidden layer

B9 = []

# bias of tau2 nodes as identity map 

for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B9.append(temp_row)

# bias matrix for alpha3, alpha4 nodes to calculate delta(x'j,i) nodes 

for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1):
            b = 0
            if q==0:
                b=(-i/eps)+1
            else:
                b=(-i/eps)
            temp_row.append(b)
        B9.append(temp_row) 
# bias matrix for beta3, beta4 nodes to calculate delta(x'j,i) nodes 

for i in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for k in range(1):
            b = 0
            if q==0:
                b=(i/eps)+1
            else:
                b=(i/eps)
            temp_row.append(b)
        B9.append(temp_row) 

# bias matrix for alpha5, alpha6 nodes to calculate delta(x'j,k) nodes 

for k in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1):
            b = 0
            if q==0:
                b=(-k/eps)+1
            else:
                b=(-k/eps)
            temp_row.append(b)
        B9.append(temp_row) 
# bias matrix for beta5, beta6 nodes to calculate delta(x'j,k) nodes 

for k in range(1, n+d+1):
  for j in range(1, d+1):
     for q in range(2):
        temp_row = []
        for i in range(1):
            b = 0
            if q==0:
                b=(k/eps)+1
            else:
                b=(k/eps)
            temp_row.append(b)
        B9.append(temp_row)
# bias matrix for alpha7, alpha8 nodes to calculate delta(x'j,x_{j+d}) nodes

for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b = 0
            if q==0:
                b=1
            temp_row.append(b)
        B9.append(temp_row)         
        
# bias matrix for beta7, beta8 nodes to calculate delta(x'j,x_{j+d}) nodes

for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            b = 0
            if q==0:
                b=1
            temp_row.append(b)
        B9.append(temp_row)               
# print('Printing B9')
# for i in B9:
#     print(i)
##################################
L9 = []  # alpha3, alpha4, beta3, beta4, alpha5, alpha6, beta5, beta6, alpha7, alpha8, beta7, beta8 nodes
for i in range(len(W9)):
    temp_row = []
    L9_i_entry = np.maximum((np.dot(W9[i], L8)+B9[i]), 0)
    L9.append(L9_i_entry)
###################################
# print('Printing alpha3, alpha4, beta3, beta4, alpha5, alpha6, beta5, beta6, alpha7, alpha8, beta7, beta8 nodes for tenth layer/ninth hidden layer')
# for i in L9:
#     print(i)
##################################
# To construct weight matrix for eleventh layer/tenth hidden layer is L10=W10*L9+B10
W10 = []
# tau2 as identity map
H31 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            if i == k and l==j:
                  w =1
            temp_row.append(w)
        H31.append(temp_row)
#####
H32 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# alpha3, alpha4
          for j in range(1, d+1):
           for q in range(2):
              w = 0
              temp_row.append(w)
        H32.append(temp_row)
##########
H33 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# beta3, beta4
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H33.append(temp_row)
##########
H34 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# alpha5, alpha6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H34.append(temp_row)
##########
H35 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# beta5, beta6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H35.append(temp_row)
##########
H36 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1): # alpha7, alpha8 nodes 
          for q in range(2):
              w = 0
              temp_row.append(w)
        H36.append(temp_row)
##########
H37 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, d+1): # beta7, beta8 nodes 
          for q in range(2):
              w = 0
              temp_row.append(w)
        H37.append(temp_row)
##########
for i in range(len(H31)):
    concatenated_row = H31[i] + H32[i]+ H33[i] + H34[i]+H35[i] + H36[i]+ H37[i]
    W10.append(concatenated_row)
#####################
# for eta6 that corresponding to e_ik
H38 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for p in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        H38.append(temp_row)
#####
H39 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4
          for k in range(1, d+1):
           for q in range(2):
              w = 0
              if i==p and j == k:
                  if q==0:
                    w = 1
                  else:
                      w= -1
              temp_row.append(w)
        H39.append(temp_row)
##########
H40 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4
         for k in range(1, d+1):
          for q in range(2):
            w = 0
            if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        H40.append(temp_row)
##########
H41 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# alpha5, alpha6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H41.append(temp_row)
##########
H42 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# beta5, beta6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H42.append(temp_row)
##########
H43 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # alpha7, alpha8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w =1
                  else:
                      w=-1
              temp_row.append(w)
        H43.append(temp_row)
##########
H44 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # beta7, beta8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w = 1
                  else:
                      w= -1
              temp_row.append(w)
        H44.append(temp_row)
##########
for i in range(len(H38)):
    concatenated_row = H38[i] + H39[i]+ H40[i] + H41[i]+H42[i] + H43[i]+ H44[i]
    W10.append(concatenated_row)
#####################
# for eta7 that corresponding to e'_i
H45 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for p in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        H45.append(temp_row)
#####
H46 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4
          for k in range(1, d+1):
           for q in range(2):
              w = 0
              if i==p and j == k:
                  if q==0:
                    w = 1
                  else:
                      w= -1
              temp_row.append(w)
        H46.append(temp_row)
##########
H47 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4
         for k in range(1, d+1):
          for q in range(2):
            w = 0
            if i==p and j == k:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        H47.append(temp_row)
##########
H48 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# alpha5, alpha6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H48.append(temp_row)
##########
H49 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):# beta5, beta6
         for j in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H49.append(temp_row)
##########
H50 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # alpha7, alpha8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w =1
                  else:
                      w=-1
              temp_row.append(w)
        H50.append(temp_row)
##########
H51 = []
for i in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # beta7, beta8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w = 1
                  else:
                      w= -1
              temp_row.append(w)
        H51.append(temp_row)
##########
for i in range(len(H45)):
    concatenated_row = H45[i] + H46[i]+ H47[i] + H48[i]+H49[i] + H50[i]+ H51[i]
    W10.append(concatenated_row)
#####################
##################
# for eta8 that corresponding to pj_ik
H52 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for p in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        H52.append(temp_row)
#####
H53 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# alpha3, alpha4
          for k in range(1, d+1):
           for q in range(2):
              w = 0
              temp_row.append(w)
        H53.append(temp_row)
##########
H54 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for p in range(1, n+d+1):# beta3, beta4
         for k in range(1, d+1):
          for q in range(2):
            w = 0
            temp_row.append(w)
        H54.append(temp_row)
##########
H55 = []
for i in range(1, n+1):
  for k in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# alpha5, alpha6
         for r in range(1, d+1):
          for q in range(2):
            w = 0
            if k==l and j == r:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        H55.append(temp_row)
##########
H56 = []
for i in range(1, n+1):
  for k in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for l in range(1, n+d+1):# beta5, beta6
         for r in range(1, d+1):
          for q in range(2):
            w = 0
            if k==l and j == r:
                if q==0:
                  w = 1
                else:
                    w= -1
            temp_row.append(w)
        H56.append(temp_row)
##########
H57 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # alpha7, alpha8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w =1
                  else:
                      w=-1
              temp_row.append(w)
        H57.append(temp_row)
##########
H58 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1): # beta7, beta8 nodes 
          for q in range(2):
              w = 0
              if j == k:
                  if q==0:
                    w = 1
                  else:
                      w= -1
              temp_row.append(w)
        H58.append(temp_row)
##########
for i in range(len(H52)):
    concatenated_row = H52[i] + H53[i]+ H54[i] + H55[i]+H56[i] + H57[i]+ H58[i]
    W10.append(concatenated_row)
#####################
# #Bias matrix for eleventh layer/tenth hidden layer

B10 = []

# bias of tau2 nodes as identity map
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B10.append(temp_row) 
# bias for eta6 that corresponding to e_il
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = -3
        temp_row.append(b)
      B10.append(temp_row)
    
# bias for eta7 that corresponding to e'_i
for i in range(1, n+d+1):
    for j in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = -3
        temp_row.append(b)
      B10.append(temp_row)

# bias for eta8 that corresponding to pj_il
for i in range(1, n+1):
  for l in range(1, n+d+1):
    for j in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = -3
        temp_row.append(b)
      B10.append(temp_row) 
####################
L10 = []  # eta6, eta7, eta8 nodes
for i in range(len(W10)):
    temp_row = []
    L10_i_entry = np.maximum((np.dot(W10[i], L9)+B10[i]), 0)
    L10.append(L10_i_entry)
###################################
# print('Printing eta6, eta7, eta8 nodes for eleventh layer/tenth hidden layer')
# for i in L10:
#     print(i)
##################################
# To construct weight matrix for twelfth layer/eleventh hidden layer is L11=W11*L10+B11
W11 = []
# tau2 as identity map
I1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            if i==k and j == l:
                 w = 1
            temp_row.append(w)
        I1.append(temp_row)
#####
I2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#   ej_il
          for p in range(1, n+d+1):
            for j in range(1, d+1):
              w = 0
              temp_row.append(w)
        I2.append(temp_row)
##########
I3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#  e'j_i
          for j in range(1, d+1):
            w = 0
            temp_row.append(w)
        I3.append(temp_row)
##########
I4 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#pj_il
          for p in range(1, n+d+1):
            for j in range(1, d+1):
             w = 0
             temp_row.append(w)
        I4.append(temp_row)
##########
for i in range(len(I1)):
    concatenated_row = I1[i] + I2[i]+ I3[i] + I4[i]
    W11.append(concatenated_row)
#####################
# # gamma2 nodes for e_ik
I5 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        I5.append(temp_row)
#####
I6 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#   ej_ik
          for l in range(1, n+d+1):
            for j in range(1, d+1):
              w = 0
              if i == p and k==l:
                 w = -1
              temp_row.append(w)
        I6.append(temp_row)
##########
I7 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  e'j_i
          for j in range(1, d+1):
            w = 0
            temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+1):#pj_il
          for p in range(1, n+d+1):
            for j in range(1, d+1):
             w = 0
             temp_row.append(w)
        I8.append(temp_row)
##########
for i in range(len(I5)):
    concatenated_row = I5[i] + I6[i]+ I7[i] + I8[i]
    W11.append(concatenated_row)
#####################
# # gamma3 nodes for e_i
I9 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        I9.append(temp_row)
#####
I10 = []
for i in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#   ej_il
          for l in range(1, n+d+1):
            for j in range(1, d+1):
              w = 0
              temp_row.append(w)
        I10.append(temp_row)
##########
I11 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  e'j_i
          for j in range(1, d+1):
            w = 0
            if i == l:
               w = -1
            temp_row.append(w)
        I11.append(temp_row)
##########
I12 = []
for i in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+1):#pj_il
          for p in range(1, n+d+1):
            for j in range(1, d+1):
             w = 0
             temp_row.append(w)
        I12.append(temp_row)
##########
for i in range(len(I9)):
    concatenated_row = I9[i] + I10[i]+ I11[i] + I12[i]
    W11.append(concatenated_row)
# ####################
# # gamma4 nodes for p_il
I13 = []
for i in range(1, n+1):
   for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        I13.append(temp_row)
#####
I14 = []
for i in range(1, n+1):
   for k in range(1, n+d+1):
        temp_row = []
        for p in range(1, n+d+1):#   ej_ik
          for l in range(1, n+d+1):
            for j in range(1, d+1):
              w = 0
              temp_row.append(w)
        I14.append(temp_row)
##########
I15 = []
for i in range(1, n+1):
   for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+d+1):#  e'j_i
          for j in range(1, d+1):
            w = 0
            temp_row.append(w)
        I15.append(temp_row)
##########
I16 = []
for i in range(1, n+1):
   for k in range(1, n+d+1):
        temp_row = []
        for l in range(1, n+1):#pj_ik
          for p in range(1, n+d+1):
            for j in range(1, d+1):
              w = 0
              if i == l and k==p:
                 w = -1
              temp_row.append(w)
        I16.append(temp_row)
##########
for i in range(len(I13)):
    concatenated_row = I13[i] + I14[i]+ I15[i] + I16[i]
    W11.append(concatenated_row)
# ####################
# #Bias matrix for twelfth layer/eleventh hidden layer

B11 = []
# bias of tau2 nodes as identity map
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
        b = 0
        temp_row.append(b)
      B11.append(temp_row) 
# bias for gamma2 that corresponding to e_ik
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
        b = 1
        temp_row.append(b)
      B11.append(temp_row)
    
# bias for gamma3 that corresponding to e'_i
for i in range(1, n+d+1):
      temp_row = []
      for k in range(1):
        b = 1
        temp_row.append(b)
      B11.append(temp_row)

# bias for gamma4 that corresponding to p_ik
for i in range(1, n+1):
  for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
          b = 1
          temp_row.append(b)
      B11.append(temp_row)
############################      
L11 = []  # gamma2, gamma3, gamma4 nodes
for i in range(len(W11)):
    temp_row = []
    L11_i_entry = np.maximum((np.dot(W11[i], L10)+B11[i]), 0)
    L11.append(L11_i_entry)
###################################
# print('Printing gamma2, gamma3, gamma4 nodes for twelfth layer/eleventh hidden layer')
# for i in L11:
#     print(i)
##################################
# To construct weight matrix for thirteenth layer/twelfth hidden layer is L12=W12*L11+B12
W12 = []
# tau2 as identity map
I17 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            if i==k and j == l:
                 w = 1
            temp_row.append(w)
        I17.append(temp_row)
#####
I18 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#   gamma2 for e_ik
          for p in range(1, n+d+1):
              w = 0
              temp_row.append(w)
        I18.append(temp_row)
##########
I19 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#  gamma3 for e'_i
            w = 0
            temp_row.append(w)
        I19.append(temp_row)
##########
I20 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):# gamma4 for p_ik
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        I20.append(temp_row)
##########
for i in range(len(I17)):
    concatenated_row = I17[i] + I18[i]+ I19[i] + I20[i]
    W12.append(concatenated_row)
#####################
# gamma5 nodes used for summation 1 to i in e_il
I21 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        I21.append(temp_row)
#####
I22 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                if k <= i and l==p:
                    w = 1
                temp_row.append(w)
            I22.append(temp_row)
##########
I23 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#  gamma3 for e'_i
            w = 0
            temp_row.append(w)
        I23.append(temp_row)
##########
I24 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):# gamma4 for p_ik
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        I24.append(temp_row)
##########
for i in range(len(I21)):
    concatenated_row = I21[i] + I22[i]+ I23[i] + I24[i]
    W12.append(concatenated_row)
#######################
# gamma6 nodes (These nodes are used for summation 1 to i e'_i)
I25 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):
          for j in range(1, n+d+1):
            w = 0
            temp_row.append(w)
        I25.append(temp_row)
#####
I26 = []
for i in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            I26.append(temp_row)
##########
I27 = []
for i in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        if k <= i:
                            w = 1
                        temp_row.append(w)
            I27.append(temp_row)
##########
I28 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):# gamma4 for p_ik
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        I28.append(temp_row)
##########
for i in range(len(I25)):
    concatenated_row = I25[i] + I26[i]+ I27[i] + I28[i]
    W12.append(concatenated_row)
#######################
# lambda1, lambda2 nodes for delta(e_ik,0)
D16 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D16.append(temp_row)

##########
D17 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                if k == i and l==p:
                    w = 1/eps
                temp_row.append(w)
            D17.append(temp_row)
##########
D18 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        temp_row.append(w)
            D18.append(temp_row)
##########
D19 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        temp_row.append(w)
            D19.append(temp_row)
##########
for i in range(len(D16)):
    concatenated_row = D16[i] + D17[i] + D18[i]+D19[i]  
    W12.append(concatenated_row)
#######################
# mu1, mu2 nodes for delta(e_ik,0)
D21 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D21.append(temp_row)
##########
D22 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                if k == i and l==p:
                    w = -1/eps
                temp_row.append(w)
            D22.append(temp_row)
##########
D23 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        temp_row.append(w)
            D23.append(temp_row)
##########
D24 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        temp_row.append(w)
            D24.append(temp_row)
##########
for i in range(len(D21)):
    concatenated_row = D21[i] + D22[i] + D23[i]+D24[i]  
    W12.append(concatenated_row)
#######################
# lambda3, lambda4 nodes for delta(e'_i,0)
D26 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D26.append(temp_row)

##########
D27 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D27.append(temp_row)
##########
D28 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        if k == i:
                            w = 1/eps
                        temp_row.append(w)
            D28.append(temp_row)
##########
D29 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        temp_row.append(w)
            D29.append(temp_row)
##########
for i in range(len(D26)):
    concatenated_row = D26[i] + D27[i] + D28[i]+D29[i]
    W12.append(concatenated_row)
#######################
# mu3, mu4 nodes for delta(e'_i,0)
D31 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D31.append(temp_row)

##########
D32 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D32.append(temp_row)
##########
D33 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        if k == i:
                            w = -1/eps
                        temp_row.append(w)
            D33.append(temp_row)
##########
D34 = []
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        temp_row.append(w)
            D34.append(temp_row)
##########
for i in range(len(D31)):
    concatenated_row = D31[i] + D32[i] + D33[i] + D34[i]
    W12.append(concatenated_row)
#######################
# gamma7 nodes for summation 1 to l p_ik
D36 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D36.append(temp_row)
##########
D37 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D37.append(temp_row)
##########
D38 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        temp_row.append(w)
            D38.append(temp_row)
##########
D39 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        if j == i and k<=l:
                            w = 1
                        temp_row.append(w)
            D39.append(temp_row)
##########
for i in range(len(D36)):
    concatenated_row = D36[i] + D37[i] + D38[i]+D39[i]  
    W12.append(concatenated_row)
#####################
# lambda5, lambda6 nodes for delta(p_ik,0)
D41 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D41.append(temp_row)

##########
D42 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D42.append(temp_row)
##########
D43 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        temp_row.append(w)
            D43.append(temp_row)
##########
D44 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        if j == i and l==k:
                            w = 1/eps
                        temp_row.append(w)
            D44.append(temp_row)
##########
for i in range(len(D41)):
    concatenated_row = D41[i] + D42[i] + D43[i]+D44[i]
    W12.append(concatenated_row)
#######################
# mu5, mu6 nodes for delta(p_ik,0)
D46 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+d+1):
              for k in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D46.append(temp_row)

##########
D47 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma2 nodes
              for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
            D47.append(temp_row)
##########
D48 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):  # gamma3 nodes
                        w = 0
                        temp_row.append(w)
            D48.append(temp_row)
##########
D49 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for j in range(1, n+1):  # gamma4 nodes
                for k in range(1, n+d+1):
                        w = 0
                        if j == i and l==k:
                            w = -1/eps
                        temp_row.append(w)
            D49.append(temp_row)
##########
for i in range(len(D46)):
    concatenated_row = D46[i] + D47[i] + D48[i]+D49[i]  
    W12.append(concatenated_row)
######################
# print("weight matrix for thirteenth layer/twelfth hidden layer")
# print(W12)
###################### 
# #Bias matrix for thirteenth layer/twelfth hidden layer

B12 = []
# bias of tau2 nodes as identity map
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
        b = 0
        temp_row.append(b)
      B12.append(temp_row) 
# # bias matrix for gamma5 nodes used for summation 1 to i e_il
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B12.append(temp_row) 
            
# bias matrix for gamma6 nodes (These nodes are used for summation 1 to i e'_i)
for i in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B12.append(temp_row)

# bias matrix for lambda1, lambda2 nodes for delta(e_il,0)
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)

# bias matrix for mu1, mu2 nodes for delta(e_il,0)
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)

# bias matrix for lambda3, lambda4 nodes for delta(e'_i,0)
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)

# bias matrix for mu3, mu4 nodes for delta(e'_i,0)
for i in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)
# bias matrix for gamma7
for i in range(1, n+1):
    for l in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B12.append(temp_row)
# bias matrix for lambda5, lambda6 nodes for delta(p_il,0)
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)

# bias matrix for mu5, mu6 nodes for delta(p_il,0)
for i in range(1, n+1):
  for l in range(1, n+d+1):
      for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B12.append(temp_row)         
############################     
L12 = []  # tau2, gamma5, gamma6, lambda1, lambda2, mu1, mu2, lambda3, lambda4, mu3, mu4, gamma7, lambda5, lambda6,mu5, mu6 nodes
for i in range(len(W12)):
    temp_row = []
    L12_i_entry = np.maximum((np.dot(W12[i], L11)+B12[i]), 0)
    L12.append(L12_i_entry)
###################################
# print('Printing tau2, gamma5, gamma6, lambda1, lambda2, mu1, mu2, lambda3, lambda4, mu3, mu4, gamma7, lambda5, lambda6,mu5, mu6 nodes for thirteenth layer/twelfth hidden layer')
# for i in L12:
#     print(i)
##################################
# To construct weight matrix for fourteenth layer/thirteenth hidden layer is L13=W13*L12+B13
W13 = []
# tau2 as identity map
I29 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                if i==k and l==j:
                    w=1
                temp_row.append(w)
        I29.append(temp_row)
##########
I30 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma5 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        I30.append(temp_row)

##########
I31 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma6 nodes
                w = 0
                temp_row.append(w)
        I31.append(temp_row)
##########
I32 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda1, lambda2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        I32.append(temp_row)
##########
I33 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu1, mu2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        I33.append(temp_row)
##########
I34 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda3, lambda4
              for q in range(2):
                w = 0
                temp_row.append(w)
        I34.append(temp_row)
##########
I35 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu3, mu4
              for q in range(2):
                w = 0
                temp_row.append(w)
        I35.append(temp_row)
##########
I36 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#gamma7
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        I36.append(temp_row)

##########
I37 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#lambda5, lambda6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        I37.append(temp_row)
##########
I38 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#mu5, mu6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        I38.append(temp_row)
##########
for i in range(len(I29)):
    concatenated_row = I29[i] + I30[i] + I31[i] + I32[i] + I33[i] + I34[i] + I35[i] + I36[i] + I37[i] + I38[i] 
    W13.append(concatenated_row)
######################
# gamma8 nodes for f_il
E11 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E11.append(temp_row)
##########
E12 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma5 nodes
          for j in range(1, n+d+1):
                w = 0
                if i==k and l==j:
                    w=B
                temp_row.append(w)
        E12.append(temp_row)

##########
E13 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma6 nodes
                w = 0
                temp_row.append(w)
        E13.append(temp_row)
##########
E14 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda1, lambda2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                if i==k and l==j:
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E14.append(temp_row)
##########
E15 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu1, mu2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                if i==k and l==j:
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E15.append(temp_row)
##########
E16 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda3, lambda4
              for q in range(2):
                w = 0
                temp_row.append(w)
        E16.append(temp_row)
##########
E17 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu3, mu4
              for q in range(2):
                w = 0
                temp_row.append(w)
        E17.append(temp_row)
##########
E18 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#gamma7
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E18.append(temp_row)

##########
E19 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#lambda5, lambda6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E19.append(temp_row)
##########
E20 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#mu5, mu6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E20.append(temp_row)
##########
for i in range(len(E11)):
    concatenated_row = E11[i] + E12[i] + E13[i] + E14[i] + E15[i] + E16[i] + E17[i] + E18[i] + E19[i] + E20[i] 
    W13.append(concatenated_row)
######################
# gamma9 nodes for f_i
E21 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E21.append(temp_row)
##########
E22 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma5 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E22.append(temp_row)

##########
E23 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma6 nodes
                w = 0
                if i==k:
                    w=B
                temp_row.append(w)
        E23.append(temp_row)
##########
E24 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda1, lambda2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E24.append(temp_row)
##########
E25 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu1, mu2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E25.append(temp_row)
##########
E26 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda3, lambda4
              for q in range(2):
                w = 0
                if i==k:
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E26.append(temp_row)
##########
E27 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu3, mu4
              for q in range(2):
                w = 0
                if i==k :
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E27.append(temp_row)
##########
E28 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#gamma7
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E28.append(temp_row)

##########
E29 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#lambda5, lambda6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E29.append(temp_row)
##########
E30 = []
for i in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#mu5, mu6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E30.append(temp_row)
##########
for i in range(len(E21)):
    concatenated_row = E21[i] + E22[i] + E23[i] + E24[i] + E25[i] + E26[i] + E27[i] + E28[i] + E29[i] + E30[i] 
    W13.append(concatenated_row)
#######################
# gamma10 nodes for q_il
E31 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E31.append(temp_row)
##########
E32 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma5 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        E32.append(temp_row)

##########
E33 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma6 nodes
                w = 0
                temp_row.append(w)
        E33.append(temp_row)
##########
E34 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda1, lambda2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E34.append(temp_row)
##########
E35 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu1, mu2
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        E35.append(temp_row)
##########
E36 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#lambda3, lambda4
              for q in range(2):
                w = 0
                temp_row.append(w)
        E36.append(temp_row)
##########
E37 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):#mu3, mu4
              for q in range(2):
                w = 0
                temp_row.append(w)
        E37.append(temp_row)
##########
E38 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#gamma7
          for j in range(1, n+d+1):
                w = 0
                if i==k and l==j:
                    w=B
                temp_row.append(w)
        E38.append(temp_row)

##########
E39 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#lambda5, lambda6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                if i==k and l==j:
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E39.append(temp_row)
##########
E40 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#mu5, mu6
          for j in range(1, n+d+1):
              for q in range(2):
                w = 0
                if i==k and l==j:
                    if q==0:
                        w=-C
                    else:
                        w=C
                temp_row.append(w)
        E40.append(temp_row)
##########
for i in range(len(E31)):
    concatenated_row = E31[i] + E32[i] + E33[i] + E34[i] + E35[i] + E36[i] + E37[i] + E38[i] + E39[i] + E40[i] 
    W13.append(concatenated_row)
#######################
# print("weight matrix for fourteenth layer/thirteenth hidden layer")
# print(W13)
#####################
#Bias matrix for fourteenth layer/thirteenth hidden layer

B13 = []
# bias of tau2 nodes as identity map
for i in range(1, n+d+1):
  for k in range(1, n+d+1):
      temp_row = []
      for l in range(1):
        b = 0
        temp_row.append(b)
      B13.append(temp_row) 
      
# bias matrix for gamma8 nodes for f_ik
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
      temp_row = []
      for k in range(1):
        b = C
        temp_row.append(b)
      B13.append(temp_row)  

# bias matrix for gamma9 nodes for f_i
for i in range(1, n+d+1):
      temp_row = []
      for k in range(1):
        b = C
        temp_row.append(b)
      B13.append(temp_row)   

# bias matrix for gamma10 nodes for q_il
for i in range(1, n+1):
  for l in range(1, n+d+1):
    temp_row = []
    for j in range(1):
        b = C
        temp_row.append(b)
    B13.append(temp_row)   
############################      
L13 = []  # tau2, gamma8, gamma9, gamma10 nodes
for i in range(len(W13)):
    temp_row = []
    L13_i_entry = np.maximum((np.dot(W13[i], L12)+B13[i]), 0)
    L13.append(L13_i_entry)
###################################
# print('Printing tau2, gamma8(f_ik), gamma9(fprime_i), gamma10(q_ik) nodes for fourteenth layer/thirteenth hidden layer')
# for i in L13:
#     print(i)
################################## 
# To construct weight matrix for fifteenth layer/fourteenth hidden layer is L14=W14*L13+B14
W14 = []
# tau2 as identity map
J1 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for j in range(1, n+d+1):
                w = 0
                if i==k and l==j:
                    w=1
                temp_row.append(w)
        J1.append(temp_row)
##########
J2 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma8 nodes
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J2.append(temp_row)

##########
J3 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+d+1):# gamma9 nodes
                w = 0
                temp_row.append(w)
        J3.append(temp_row)
##########
J4 = []
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
        temp_row = []
        for k in range(1, n+1):#gamma10
          for j in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        J4.append(temp_row)
##########
for i in range(len(J1)):
    concatenated_row = J1[i] + J2[i] + J3[i] + J4[i] 
    W14.append(concatenated_row)
######################
# psi1 psi2 nodes(These nodes correspond to [f_(i+j-1)l>=iB])
J5 = []
for i in range(1, n+1):
  for k in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for l in range(1, n+d+1):# tau2 nodes
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J5.append(temp_row)
##########
J6 = []
for i in range(1, n+1):
  for k in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for r in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  if i+j-1 == r and p == k:
                      w = 1/eps
                  temp_row.append(w)
            J6.append(temp_row)
##########
J7 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  temp_row.append(w)
            J7.append(temp_row)
##########
J8 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J8.append(temp_row)
##########
for i in range(len(J5)):
    concatenated_row = J5[i] + J6[i] + J7[i] + J8[i] 
    W14.append(concatenated_row)
######################
# rho1 rho2 nodes(These nodes correspond to [-f_(i+j-1)>=-(iB+1)])
J9 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J9.append(temp_row)
##########
J10 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  if i+j-1 == k and p == l:
                      w = -1/eps
                  temp_row.append(w)
            J10.append(temp_row)
##########
J11 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  temp_row.append(w)
            J11.append(temp_row)
##########
J12 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J12.append(temp_row)
##########
for i in range(len(J9)):
    concatenated_row = J9[i] + J10[i] + J11[i] + J12[i] 
    W14.append(concatenated_row)
# print(W14)
#######################
# psi3 psi4 nodes(These nodes correspond to [f_(i+j-1)>=iB])
J13 = []
for i in range(1, n+1):
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J13.append(temp_row)
##########
J14 = []
for i in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  temp_row.append(w)
            J14.append(temp_row)
##########
J15 = []
for i in range(1, n+1): 
      for j in range(1, d+2):
        for q in range(2):
            temp_row = []
            for l in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  if i+j-1 == l:
                      w = 1/eps
                  temp_row.append(w)
            J15.append(temp_row)

##########
J16 = []
for i in range(1, n+1):
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J16.append(temp_row)
##########
for i in range(len(J13)):
    concatenated_row = J13[i] + J14[i] + J15[i] + J16[i] 
    W14.append(concatenated_row)
#######################
# rho3 rho4 nodes(These nodes correspond to [-f_(i+j-1)>=-(iB+1)])
J17 = []
for i in range(1, n+1):
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J17.append(temp_row)
##########
J18 = []
for i in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  temp_row.append(w)
            J18.append(temp_row)
##########
J19 = []
for i in range(1, n+1): 
      for j in range(1, d+2):
        for q in range(2):
            temp_row = []
            for l in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  if i+j-1 == l:
                      w = -1/eps
                  temp_row.append(w)
            J19.append(temp_row)
##########
J20 = []
for i in range(1, n+1):
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J20.append(temp_row)
##########
for i in range(len(J17)):
    concatenated_row = J17[i] + J18[i] + J19[i] + J20[i] 
    W14.append(concatenated_row)
# #######################
# psi5 psi6 nodes(These nodes correspond to [q_i(l+j-1)>=lB])
J21 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J21.append(temp_row)

##########
J22 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  temp_row.append(w)
            J22.append(temp_row)

##########
J23 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  temp_row.append(w)
            J23.append(temp_row)

##########
J24 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  if l+j-1 == p and i == k:
                      w = 1/eps
                  temp_row.append(w)
            J24.append(temp_row)
##########
for i in range(len(J21)):
    concatenated_row = J21[i] + J22[i] + J23[i] + J24[i] 
    W14.append(concatenated_row)
# print(W14)
#######################
# rho5 rho6 nodes(These nodes correspond to [-q_i(l+j-1)>=lB+1])
J25 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):
                  w = 0
                  temp_row.append(w)
            J25.append(temp_row)

##########
J26 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):
              for p in range(1, n+d+1):# gamma 8 nodes
                  w = 0
                  temp_row.append(w)
            J26.append(temp_row)

##########
J27 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+d+1):# gamma9 nodes
                  w = 0
                  temp_row.append(w)
            J27.append(temp_row)

##########
J28 = []
for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1, n+1):# gamma10 nodes
              for p in range(1, n+d+1):
                  w = 0
                  if l+j-1 == p and i == k:
                      w = -1/eps
                  temp_row.append(w)
            J28.append(temp_row)
##########
for i in range(len(J25)):
    concatenated_row = J25[i] + J26[i] + J27[i] + J28[i] 
    W14.append(concatenated_row)
#######################
# print("weight matrix for fifteenth layer/fourteenth hidden layer")
# print(W14)
#####################
#Bias matrix for fifteenth layer/fourteenth hidden layer
B14 = []

# bias matrix tau2 nodes as identity map
for i in range(1, n+d+1):
  for l in range(1, n+d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B14.append(temp_row)   
         
# bias matrix for psi1 psi2 nodes(These nodes correspond to [f_(i+j-1)l>=iB])

for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b = -i*B / eps+1
                else:    
                    b = -i*B / eps 
                temp_row.append(b)
            B14.append(temp_row)
            
# bias matrix for rho1 rho2 nodes(These nodes correspond to [f_(i+j-1)l>=iB])

for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b = (i*B+1) / eps+1
                else:    
                    b = (i*B+1) / eps
                temp_row.append(b)
            B14.append(temp_row)

# bias matrix for psi3 psi4 nodes(These nodes correspond to [f_(i+j-1)>=iB])

for i in range(1, n+1):
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b = -i*B / eps+1
                else:    
                    b = -i*B / eps
                temp_row.append(b)
            B14.append(temp_row)

# bias matrix for rho3 rho4 nodes(These nodes correspond to [f_(i+j-1)l>=iB+1])

for i in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b = (i*B+1) / eps+1
                else:    
                    b = (i*B+1) / eps
                temp_row.append(b)
            B14.append(temp_row)
            
# bias matrix for psi5 psi6 nodes(These nodes correspond to [-q_i(l+j-1)>=lB]))

for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b = (-l*B / eps)+1
                else:    
                    b = -l*B / eps 
                temp_row.append(b)
            B14.append(temp_row)
            
# bias matrix for rho5 rho6 nodes(These nodes correspond to [-q_i(l+j-1)>=lB+1]))

for i in range(1, n+1):
  for l in range(1, n+1): 
      for j in range(1, d+2): 
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q==0:
                    b =( (l*B+1) / eps)+1
                else:    
                    b = (l*B+1) / eps
                temp_row.append(b)
            B14.append(temp_row)            
#################################
# print('Printing B14')
# for i in B14:
#     print(i)
##################################
L14 = []  # psi1, psi2, rho1, rho2, psi3, psi4, rho3, rho4, psi5, psi6, rho5, rho6 nodes
for i in range(len(W14)):
    temp_row = []
    L14_i_entry = np.maximum((np.dot(W14[i], L13)+B14[i]), 0)
    L14.append(L14_i_entry)
##################################
# print('Printing psi1, psi2, rho1, rho2, psi3, psi4, rho3, rho4, psi5, psi6, rho5, rho6 nodes for fifteenth layer/fourteenth hidden layer')
# for i in L14:
#     print(i) 
###########################
# To construct weight matrix for Sixteenth layer/fifteenth hidden layer is L15=W15*L14+B15
W15 = [] 
# zeta1 nodes corresponding to h^j_ik
G1 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for p in range(1, n+d+1):
                w = 0
                if p == l and i+j-1==k:
                        w = 1
                temp_row.append(w)
        G1.append(temp_row)

##########
G2 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# psi1 psi2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == p and l==r and j==k:
                      if q==0:
                        w = C
                      else:
                        w=-C
                    temp_row.append(w)
        G2.append(temp_row)
##########
G3 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# rho1 rho2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == p and l==r and j==k:
                      if q==0:
                        w = C
                      else:
                        w=-C
                    temp_row.append(w)
        G3.append(temp_row)
##########
G4 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G4.append(temp_row)
##########
G5 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G5.append(temp_row)
##########
G6 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G6.append(temp_row)
##########
G7 = []
for i in range(1, n+1):
  for l in range(1, n+d+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G7.append(temp_row)
##########
for i in range(len(G1)):
    concatenated_row = G1[i] + G2[i] + G3[i] + G4[i] + G5[i]  + G6[i] + G7[i] 
    W15.append(concatenated_row)
#######################
# zeta2 nodes corresponding to h'^j_i
G8 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        G8.append(temp_row)

##########
G9 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# psi1 psi2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G9.append(temp_row)
##########
G10 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# rho1 rho2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G10.append(temp_row)
##########
G11 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for l in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == l and j==k:
                      if q==0:
                        w = C
                      else:
                        w=-C
                    temp_row.append(w)
        G11.append(temp_row)
##########
G12 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == p and j==k:
                      if q==0:
                        w = C
                      else:
                        w=-C
                    temp_row.append(w)
        G12.append(temp_row)
##########
G13 = []
for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G13.append(temp_row)
##########
G14 = []
for i in range(1, n+1): 
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G14.append(temp_row)
##########
for i in range(len(G8)):
    concatenated_row = G8[i] + G9[i] + G10[i] + G11[i] + G12[i]  + G13[i] + G14[i] 
    W15.append(concatenated_row)
# print(W15)    
############################
# eta10 nodes for gamma^j_ik  
G15 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1, n+d+1):# tau2 nodes
          for p in range(1, n+d+1):
                w = 0
                temp_row.append(w)
        G15.append(temp_row)

##########
G16 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# psi1 psi2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G16.append(temp_row)
##########
G17 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):# rho1 rho2 nodes
          for r in range(1, n+d+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G17.append(temp_row)
##########
G18 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G18.append(temp_row)
##########
G19 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    temp_row.append(w)
        G19.append(temp_row)
##########
G20 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == p and l==r and j==k:
                      if q==0:
                        w = 1
                      else:
                        w=-1
                    temp_row.append(w)
        G20.append(temp_row)
##########
G21 = []
for i in range(1, n+1): 
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1): 
              for k in range(1, d+2): 
                for q in range(2):
                    w = 0
                    if i == p and l==r and j==k:
                      if q==0:
                        w = 1
                      else:
                        w=-1
                    temp_row.append(w)
        G21.append(temp_row)
##########
for i in range(len(G15)):
    concatenated_row = G15[i] + G16[i] + G17[i] + G18[i] + G19[i]  + G20[i] + G21[i] 
    W15.append(concatenated_row)
############################
# print("weight matrix for Sixteenth layer/fifteenth hidden layer")
# print(W15)
######################
#Bias matrix for Sixteenth layer/fifteenth hidden layer

B15 = []

# bias matrix for zeta1 nodes corresponding to h^j_ik

for i in range(1, n+1):
  for l in range(1, n+d+1): 
    for j in range(1, d+2):
      temp_row = []
      for k in range(1):
          b = -2*C
          temp_row.append(b)
      B15.append(temp_row)
        
# bias matrix for zeta2 nodes corresponding to h'^j_i

for i in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1):
            b = U[i+j-2]-2*C
            temp_row.append(b)
        B15.append(temp_row)

# bias matrix for eta10 nodes 

for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1):
            b=-1
            temp_row.append(b)
        B15.append(temp_row)              
# print('Printing B15')
# for i in B15:
#     print(i)
##################################
L15 = []  # zeta 1, zeta 2, eta10 nodes
for i in range(len(W15)):
    temp_row = []
    L15_i_entry = np.maximum((np.dot(W15[i], L14)+B15[i]), 0)
    L15.append(L15_i_entry)
#################################
# print('Printing zeta 1, zeta 2, eta10 nodes for Sixteenth layer/fifteenth hidden layer')
# for i in L15:
#     print(i)
####################
# To construct weight matrix for Seventeenth layer/Sixteenth hidden layer is L16=W16*L15+B16
W16 = []
# zeta3 nodes
H1 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
          temp_row = []
          for p in range(1, n+1):# zeta1 nodes
            for r in range(1, n+d+1): 
                for j in range(1, d+2):
                    w = 0
                    if i == p and l==r:
                        w = 1
                    temp_row.append(w)
          H1.append(temp_row)
##########
H2 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
          temp_row = []          
          for p in range(1, n+1):# zeta2 nodes
                for j in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H2.append(temp_row)
##########
H3 = []
for i in range(1, n+1):
  for l in range(1, n+d+1):
          temp_row = []          
          for p in range(1, n+1):# eta10 nodes
            for r in range(1, n+1):
                for j in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H3.append(temp_row)
##########
for i in range(len(H1)):
    concatenated_row = H1[i] + H2[i] + H3[i] 
    W16.append(concatenated_row)
# print(W16)
######################
# zeta4 nodes 
H4 = []
for i in range(1, n+1):
          temp_row = []
          for p in range(1, n+1):# zeta1 nodes
            for r in range(1, n+d+1): 
                for j in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H4.append(temp_row)
##########
H5 = []
for i in range(1, n+1):
          temp_row = []          
          for p in range(1, n+1):# zeta2 nodes
                for j in range(1, d+2):
                    w = 0
                    if i == p:
                        w = 1
                    temp_row.append(w)
          H5.append(temp_row)
##########
H6 = []
for i in range(1, n+1):
          temp_row = []          
          for p in range(1, n+1):# eta10 nodes
            for r in range(1, n+1):
                for j in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H6.append(temp_row)
##########
for i in range(len(H4)):
    concatenated_row = H4[i] + H5[i] + H6[i] 
    W16.append(concatenated_row)
# print(W16)
######################
# eta10 nodes as identity map 
H7 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
          temp_row = []
          for p in range(1, n+1):# zeta1 nodes
            for r in range(1, n+d+1): 
                for j in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H7.append(temp_row)
##########
H8 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
          temp_row = []          
          for p in range(1, n+1):# zeta2 nodes
                for k in range(1, d+2):
                    w = 0
                    temp_row.append(w)
          H8.append(temp_row)
##########
H9 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
          temp_row = []          
          for p in range(1, n+1):# eta10 nodes
            for r in range(1, n+1):
                for k in range(1, d+2):
                    w = 0
                    if i==p and l==r and j==k:
                        w=1
                    temp_row.append(w)
          H9.append(temp_row)
##########
for i in range(len(H7)):
    concatenated_row = H7[i] + H8[i] + H9[i] 
    W16.append(concatenated_row)
# print(W16)
######################
# print("weight matrix for seventeenth layer/sixteenth hidden layer")
# print(W16)
#####################
# #Bias matrix for seventeenth layer/sixteenth hidden layer

#B16 is zero matrix
##################################
L16 = []  # zeta3, zeta4, eta10 nodes
for i in range(len(W16)):
    temp_row = []
    L16_i_entry = np.maximum((np.dot(W16[i], L15)), 0)
    L16.append(L16_i_entry)
############################
# print('Printing zeta3, zeta4, eta10 nodes for seventeenth layer/sixteenth hidden layer')
# for i in L16:
#     print(i)
############################
# To construct weight matrix for eighteenth layer/seventeenth hidden layer is L17=W17*L16+B17
W17 = []
# zeta4 nodes as identity map 
H10 = []
for i in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
          for p in range(1, n+d+1):
             w = 0
             temp_row.append(w)
        H10.append(temp_row)
##########
H11 = []
for i in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
              w = 0
              if i==k:
                w = 1
              temp_row.append(w)
        H11.append(temp_row)
##########
H12 = []
for i in range(1, n+1):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1):
             for k in range(1, d+2):
              w = 0
              temp_row.append(w)
        H12.append(temp_row)
##########
for i in range(len(H10)):
    concatenated_row = H10[i] + H11[i]+ H12[i] 
    W17.append(concatenated_row)    
#######################
# zeta5 nodes 
H13 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1, n+1):
          for p in range(1, n+d+1):
             w = 0
             if i==k and l+j-1==p:
               w = 1
             temp_row.append(w)
        H13.append(temp_row)
##########
H14 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1, n+1):
              w = 0
              temp_row.append(w)
        H14.append(temp_row)
##########
H15 = []
for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for p in range(1, n+1):
          for r in range(1, n+1):
             for k in range(1, d+2):
              w = 0
              if i==p and l==r and j==k:
                  w=C
              temp_row.append(w)
        H15.append(temp_row)
##########
for i in range(len(H13)):
    concatenated_row = H13[i] + H14[i]+ H15[i] 
    W17.append(concatenated_row)    
#######################  
# print("weight matrix for eighteenth layer/seventeenth hidden layer")
# print(W17)
#####################
# #Bias matrix for eighteenth layer/seventeenth hidden layer

B17 = []

# bias matrix for zeta4 nodes

for i in range(1, n+1):
      temp_row = []
      for k in range(1):
          b = 0
          temp_row.append(b)
      B17.append(temp_row)

# bias matrix for zeta5 nodes 

for i in range(1, n+1):
  for l in range(1, n+1):
      for j in range(1, d+2):
        temp_row = []
        for k in range(1):
            b=-C
            temp_row.append(b)
        B17.append(temp_row)              
# print('Printing B17')
# for i in B17:
#     print(i)
##################################
L17 = []  # zeta5 nodes
for i in range(len(W17)):
    temp_row = []
    L17_i_entry = np.maximum((np.dot(W17[i], L16)+B17[i]), 0)
    L17.append(L17_i_entry)
###################################
# print('Printing zeta5 nodes for eighteenth layer/seventeenth hidden layer')
# for i in L17:
#     print(i)
##################################
# To construct weight matrix for ninteenth layer/eighteenth hidden layer is L18=W18*L17+B18
W18 = []
# zeta4 nodes 
H16 = []
for i in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
             w = 0
             if i==k:
                 w=1
             temp_row.append(w)
        H16.append(temp_row)
##########
H17 = []
for i in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
          for l in range(1, n+1):
              for j in range(1, d+2):
                w = 0
                temp_row.append(w)
        H17.append(temp_row)
##########
for i in range(len(H16)):
    concatenated_row = H16[i] + H17[i] 
    W18.append(concatenated_row)    
######################
# zeta6 nodes 
H18 = []
for i in range(1, n+1):
    for l in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
             w = 0
             temp_row.append(w)
        H18.append(temp_row)
##########
H19 = []
for i in range(1, n+1):
    for l in range(1, n+1):
        temp_row = []
        for k in range(1, n+1):
          for r in range(1, n+1):
              for j in range(1, d+2):
                w = 0
                if i==k and l==r:
                    w=1
                temp_row.append(w)
        H19.append(temp_row)
##########
for i in range(len(H18)):
    concatenated_row = H18[i] + H19[i] 
    W18.append(concatenated_row) 
#####################
# print("weight matrix for ninteenth layer/eighteenth hidden layer")
# print(W18)
#####################
# #Bias matrix for ninteenth layer/eighteenth hidden layer

#B18 is a zero matrix      
################################## 
Y = []  # output
for i in range(len(W18)):
    temp_row = []
    Y_i_entry = np.maximum((np.dot(W18[i], L17)), 0)
    Y.append(Y_i_entry)
#################################
# print('Printing output nodes for eleventh layer/tenth hidden layer')
# for i in Y:
#     print(i)
############
# Flatten the output
flat_Y = [y[0] for y in Y]

# Extract L' (first n values)
L_star = flat_Y[:n]

# Extract A' (remaining n*n values and reshape to n x n matrix)
A_star_flat = flat_Y[n:]
A_star = [A_star_flat[i * n : (i + 1) * n] for i in range(n)]
#####################################
# U_prime = [int(x) for x in L_star]
# print("U' that is output label matrix including B:", U_prime)
# ######################################
# # output padded adjacency matrix V' (with 1000s as padding)
# keep_rows = [i for i in range(n) if any(val != B for val in A_star[i])]
# keep_cols = [j for j in range(n) if any(A_star[i][j] != B for i in range(n))]

# # Step 2: Build the new adjacency matrix with integers
# V_prime = [
#     [int(A_star[i][j]) for j in keep_cols]
#     for i in keep_rows
# ]

# # Step 3: Print the cleaned matrix
# print("V' that is padded adjacency matrix of output graph is:")
# for row in V_prime:
#     print(row)
# #################################    
# Remove B and convert remaining to integers
L_filtered = [int(x) for x in L_star if x != B]

print("L' that is label matrix of output graph is:", L_filtered)
#########################################
# output adjacency matrix A' (with 1000s as padding)
keep_rows = [i for i in range(n) if any(val != B for val in A_star[i])]
keep_cols = [j for j in range(n) if any(A_star[i][j] != B for i in range(n))]

# Step 2: Build the new adjacency matrix with integers
A_reduced = [
    [int(A_star[i][j]) for j in keep_cols]
    for i in keep_rows
]

# Step 3: Print the cleaned matrix
print("A' that is adjacency matrix of output graph is:")
for row in A_reduced:
    print(row)
#########################################