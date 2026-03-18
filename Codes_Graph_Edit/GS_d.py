# -*- coding: utf-8 -*-
"""
Created on Tueday, August 20,2025 at 1:43:00

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
L = [3, 5, 4, 2, 4]
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

#To construct weight matrix for second layer/first hidden layer is L1=W1*X+B1
W1 = []

#top(#for psi and rho nodes)      
for k in range(d):  #(psi nodes)
    for l in range(d):
        for q in range(2):
            temp_row = []
            for j in range(2 * d):
                w = 0
                if l == j and k != l:
                    w = 1 / eps
                if k == j and k != l:
                    w = -1 / eps    
                temp_row.append(w)
            W1.append(temp_row)
##
for k in range(d):  #(rho nodes)
    for l in range(d):
        for q in range(2):
            temp_row = []
            for j in range(2 * d):
                w = 0
                if l == j and k != l:
                    w = -1 / eps
                if k == j and k != l:
                    w = 1 / eps    
                temp_row.append(w)
            W1.append(temp_row)
# bottom(for identity map of x_j)
for k in range(2 * d):
    temp_row = []
    for j in range(2 * d):
        w = 0
        if k == j:
            w = 1
        temp_row.append(w)
    W1.append(temp_row)   
# print('weight matrix of second layer/first hidden layer')    
# for i in W1:
#     print(i)
    
#Bias matrix for second layer/first hidden layer

B1 = []
# (bias of psi nodes)

for k in range(d):
    for l in range(d):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q==0:
                    b = 1
                else:
                    b=0
                temp_row.append(b)
            B1.append(temp_row)

# (bias of rho nodes)

for k in range(d):
    for l in range(d):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q==0:
                    b = 1
                else:
                    b=0
                temp_row.append(b)
            B1.append(temp_row)

#(bias of x_j)

for l in range(2 * d):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B1.append(temp_row)
    
L1 = [] # psi and rho nodes
for i in range(len(W1)):
    L1_i_entry =np.maximum( (np.dot(W1[i], X)+B1[i]),0)   
    L1.append(L1_i_entry)
    # print('this is index i:', i)
    # print('this is the value L1[i]:', L1_i_entry)
    
# print('psi, rho and x_{j} nodes for second layer/first hidden layer')
# for i in L1:
#     print(i)
##################
#To construct weight matrix for third layer/second hidden layer is L2=W2*L1+B2
W2 = [] 
A1=[]    
for j in range(d):  #(x'_j nodes)
    temp_row = []
    for k in range(d):  #(psi nodes)
       for l in range(d):
           for q in range(2):
                w = 0
                if l == j and k < l:
                 if q==0:
                    w = -C 
                 else:
                    w= C     
                temp_row.append(w)
    A1.append(temp_row)
##
A2=[]    
for j in range(d):  #(x'_j nodes)
    temp_row = []
    for k in range(d):  #(rho nodes)
       for l in range(d):
           for q in range(2):
                w = 0
                if l == j and k < l:
                 if q==0:
                    w = -C 
                 else:
                    w= C 
                temp_row.append(w)
    A2.append(temp_row)
##
A3=[]    
for j in range(d):  #(x'_j nodes)
    temp_row = []
    for k in range(d):  #(x_j nodes)
      w = 0
      if k == j:
          w = 1 
      temp_row.append(w)
    A3.append(temp_row)  
#print(W2)    
A4=[]
for j in range(d):#(x'_j nodes)
    temp_row = []
    for k in range(d):#(x_j nodes)
        w = 0
        temp_row.append(w)
    A4.append(temp_row)
##
for i in range(len(A1)):
    concatenated_row = A1[i] + A2[i] + A3[i] + A4[i]
    W2.append(concatenated_row) 
    ############
A5=[]    
for j in range(d):  #(x_j+d nodes)
    temp_row = []
    for k in range(d):  #(psi nodes)
       for l in range(d):
           for q in range(2):
                w = 0     
                temp_row.append(w)
    A5.append(temp_row)
##
A6=[]    
for j in range(d):  #(x_j+d nodes)
    temp_row = []
    for k in range(d):  #(rho nodes)
       for l in range(d):
           for q in range(2):
                w = 0
                temp_row.append(w)
    A6.append(temp_row)
##
A7=[]    
for j in range(d):  #(x_j+d nodes)
    temp_row = []
    for k in range(d):  #(x_j nodes)
      w = 0 
      temp_row.append(w)
    A7.append(temp_row)      
A8=[]#(for identity map of x_j+d)
for j in range(d):#(x'_j nodes)
    temp_row = []
    for k in range(d):
        w = 0
        if j==k:
            w=1
        temp_row.append(w)
    A8.append(temp_row)
##
for i in range(len(A5)):
    concatenated_row = A5[i] + A6[i] + A7[i] + A8[i]
    W2.append(concatenated_row)     
# print('weight matrix for third layer/second hidden layer')    
# for i in W2:
#     print(i)
    
#Bias matrix for third layer/second hidden layer 

B2 = []
# (bias of x' nodes)

for j in range(d):
    temp_row = []
    for k in range(1):
        b = C*(j-1+1)
        temp_row.append(b)
    B2.append(temp_row)

# (bias of x_j+d nodes)

for j in range(d):
   temp_row = []
   for j in range(1):
       b = 0
       temp_row.append(b)
   B2.append(temp_row)
#print(B2)    
L2 = [] # x' nodes
for i in range(len(W2)):
    L2_i_entry =np.maximum( (np.dot(W2[i], L1)+B2[i]),0)   
    L2.append(L2_i_entry)
    # print('this is index i:', i)
    # print('this is the value L2[i]:', L2_i_entry)
#####################################    
# print('prime_{j} nodes of third layer/second hidden layer')
# for i in L2:
#     print(i)
##################
#To construct weight matrix for fourth layer/third hidden layer is L3=W3*L2+B3
W3 = []
# top
for l in range(n):  #(alpha nodes)
    for k in range(d):
        for q in range(2):
            temp_row = []
            for j in range(2 * d):
                w = 0
                if k == j:
                    w = 1 / eps
                temp_row.append(w)
            W3.append(temp_row)
##       
for l in range(n):  #(beta nodes)
    for k in range(d):
        for q in range(2):
            temp_row = []
            for j in range(2 * d):
                w = 0
                if k == j:
                    w = -1 / eps
                temp_row.append(w)
            W3.append(temp_row)
##
#print(W3)
# bottom
for k in range(d, 2 * d):
    temp_row = []
    for j in range(2 * d):
        w = 0
        if k == j:
            w = 1
        temp_row.append(w)
    W3.append(temp_row)
# print('weight matrix for fourth layer/third hidden layer')    
# for i in W3:
#     print(i)
    
#Bias matrix for fourth layer/third hidden layer

B3 = []
# top1(bias of alpha for k in range(0,d))

for l in range(n):
    for k in range(d):
        for m in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if m==0:
                    b = -(l+1) / eps +1
                else:
                    b=(-(l+1) / eps)
                temp_row.append(b)
            B3.append(temp_row)

# (bias of beta for k in range(0,d))

for l in range(n):
    for k in range(d):
        for m in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if m==0:
                    b = (l+1) / eps +1
                else:
                    b=((l+1) / eps)
                temp_row.append(b)
            B3.append(temp_row)

#bottom(bias of x_j+d )

for l in range(d, 2 * d):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B3.append(temp_row)
    
L3 = [] # alpha and beta nodes
for i in range(len(W3)):
    L3_i_entry =np.maximum( (np.dot(W3[i], L2)+B3[i]),0)   
    L3.append(L3_i_entry)
    # print('this is index i:', i)
    # print('this is the value L3[i]:', L3_i_entry)
    
# print('Alpha^11, Alpha^12,Beta^11,Beta^12 and x_{j+d} nodes of fourth layer/third hidden layer')
# for i in L3:
#     print(i)
    
#To construct weight matrix for fifth layer/fourth hidden layer is L4=W4*L3+B4
W4 = []
#
C1 = []# calculate lambda and alpha nodes
for i in range(n):
    temp_row = []
    for l in range(n):
       for k in range(d):
          for q in range(2):
              w = 0
              if i == l:
                 if q == 0:
                    w = -C
                 else: # q = 1
                    w = C
              temp_row.append(w)
    C1.append(temp_row)
##
C2 = []# calculate lambda and beta nodes
for i in range(n):
    temp_row = []
    for l in range(n):
       for k in range(d):
          for q in range(2):
              w = 0
              if i == l:
                 if q == 0:
                    w = -C
                 else: # q = 1
                    w = C
              temp_row.append(w)
    C2.append(temp_row)  
##
C3 = []# calculate lambda and x_j+d nodes
for i in range(n):
    temp_row = []
    for k in range(d, 2 * d):
        w = 0
        temp_row.append(w)
    C3.append(temp_row)          
##
for i in range(len(C1)):
    concatenated_row = C1[i] + C2[i] + C3[i]
    W4.append(concatenated_row)
# print("weight matrix for fifth layer/fourth hidden layer")
# print(W4)
#
C4 = []# calculate mu and alpha nodes
for i in range(n):
    for j in range(d):
      temp_row = []
      for l in range(n):
         for k in range(d):
            for q in range(2):
              w = 0
              if i == l and j==k:
                 if q == 0:
                    w = C
                 else: # q = 1
                    w = -C
              temp_row.append(w)
      C4.append(temp_row)
##
C5 = []# calculate mu and beta nodes
for i in range(n):
    for j in range(d):
      temp_row = []
      for l in range(n):
         for k in range(d):
            for q in range(2):
              w = 0
              if i == l and j==k:
                 if q == 0:
                    w = C
                 else: # q = 1
                    w = -C
              temp_row.append(w)
      C5.append(temp_row) 
##
C6 = []# calculate mu and x_j+d nodes
for i in range(n):
    for j in range(d):
      temp_row = []
      for k in range(d, 2 * d):
         w = 0
         if k==j+d:
            w=1
         temp_row.append(w)
      C6.append(temp_row)          
##
for i in range(len(C4)):
    concatenated_row = C4[i] + C5[i] + C6[i]
    W4.append(concatenated_row)
# print("weight matrix for fifth layer/fourth hidden layer")
# print(W4)
##
# Bias matrix for fifth layer/fourth hidden layer        
B4 = [] 
for i in range(n):
    temp_row = []
    for l in range(1):
       b = L[i]+d*C
       temp_row.append(b)
    B4.append(temp_row) 
##
for i in range(n):
   for j in range(d): 
      temp_row = []
      for l in range(1):
         b = -2*C
         temp_row.append(b)
      B4.append(temp_row)    
# print("bias for fifth layer/fourth hidden layer")
# for i in B4:
#     print (i)        

L4 = [] # Lambda and Mu nodes
for i in range(len(W4)):
    L4_i_entry =np.maximum( (np.dot(W4[i], L3)+B4[i]),0)   
    L4.append(L4_i_entry)
    #print('this is index i:', i)
    #print('this is the value L4[i]:', L4_i_entry)    
# print('Printing Lambda and Mu nodes of fifth layer/fourth hidden layer')
# for i in L4:
#     print(i) 
#
#To construct weight matrix for sixth layer/fifth hidden layer is L5=W5*L4+B5
W5 = []
#
D1=[]
for i in range(n):
    temp_row = []
    for l in range(n):
       w = 0
       if i == l:
          w = 1
       temp_row.append(w)
    D1.append(temp_row)
##
D2=[]
for i in range(n):
    temp_row = []
    for l in range(n):
       for j in range (d):
         w = 0
         temp_row.append(w)
    D2.append(temp_row)    
#print(D2)
##
for i in range(len(D1)):
    concatenated_row = D1[i] + D2[i]
    W5.append(concatenated_row)

##########
D3=[]
for i in range(n):
    temp_row = []
    for l in range(n):
       w = 0
       temp_row.append(w)
    D3.append(temp_row)
##
D4=[]
for i in range(n):
    temp_row = []
    for l in range(n):
       for j in range (d):
         w = 0
         if i==l:
             w=1
         temp_row.append(w)
    D4.append(temp_row)    
#print(D4)
##
for i in range(len(D3)):
    concatenated_row = D3[i] + D4[i]
    W5.append(concatenated_row)
# print("weight for sixth layer/fifth hidden layer")
# print(W5)
##
#Bias matrix for sixth layer/fifth hidden layer
B5 = []

for i in range(n):
    temp_row = []
    for l in range(1):
        b = 0
        temp_row.append(b)
    B5.append(temp_row)
##
for i in range(n):
    temp_row = []
    for l in range(1):
        b = 0
        temp_row.append(b)
    B5.append(temp_row)    

#print("bias for sixth layer/fifth hidden layer")
#for i in B5:
#    print(i)            
    
L5 = [] # gamma and lambda nodes
for i in range(len(W5)):
    L5_i_entry = np.maximum((np.dot(W5[i], L4) +B5[i]),0)   
    L5.append(L5_i_entry)
    #print('this is index i:', i)
    #print('this is the value L5[i]:', L5_i_entry)
# print("Lambda and Gamma nodes for sixth layer/fifth hidden layer")  
# for i in L5:
#     print(i)   
#To construct weight matrix for seventh layer/sixth hidden layer is L6=W6*L5+B6  
W6 = [] 
##
E1=[]
for i in range(n):
    temp_row = []
    for l in range(n):
        w = 0
        if l==i:
          w = 1
        temp_row.append(w)
    E1.append(temp_row)
##
E2=[]
for i in range(n):
    temp_row = []
    for l in range(n):
        w = 0
        if l==i:
          w = 1
        temp_row.append(w)
    E2.append(temp_row)    
for i in range(len(E1)):
    concatenated_row = E1[i] + E2[i]
    W6.append(concatenated_row)
# print("weight for output Layer")
# print(W6)

#Bias matrix for output layer

B6 = []
for i in range(n):
    temp_row = []
    for l in range (1):
        b = 0
        temp_row.append(b)
    B6.append(temp_row)

Y = [] # Output nodes
for i in range(len(W6)):
    Y_i_entry = np.maximum((np.dot(W6[i], L5) +B6[i]),0)   
    Y.append(Y_i_entry)
    # print('this is index i:', i)
    # print('this is the value Y[i]:', Y_i_entry)

values = [int(array[0]) for array in Y]


print("L' that is the Label matrix of output graph:", values) 
###############
print("A' that is adjacency matrix of output graph is:")
for row in A:
    print(row)


