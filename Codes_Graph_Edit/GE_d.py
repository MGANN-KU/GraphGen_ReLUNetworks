# -*- coding: utf-8 -*-
"""
Created on Wed Oct 1,2025 at 11:43:00

@author: Ghafoor

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
#import torch
import gc


import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
### taken from Graph-uni_30_11_2025
from layers.layer_1 import build_W1, build_B1
from layers.layer_2 import build_W2, build_B2
from layers.layer_3 import build_W3
from layers.layer_4 import build_W4, build_B4
from layers.layer_5_b import build_W5, build_B5
from layers.layer_6 import build_W6, build_B6
from layers.layer_7 import build_W7, build_B7
from layers.layer_8_b import build_W8, build_B8
from layers.layer_9 import build_W9
from layers.layer_10 import build_W10, build_B10
from layers.layer_11 import build_W11, build_B11
from layers.layer_12 import build_W12
from layers.layer_13_b import build_W13, build_B13
from layers.layer_14 import build_W14, build_B14
#from layers.layer_17 import build_W17, build_B17

##### Taken from Graph-uni_3_12_2025_b
from layers.layer_19 import build_W19, build_B19
from layers.layer_20_c import build_W20, build_B20
from layers.layer_21_b import build_W21, build_B21
from layers.layer_22_e import build_W22, build_B22
from layers.layer_23 import build_W23, build_B23
from layers.layer_24_b import build_W24, build_B24
from layers.layer_25 import build_W25, build_B25
from layers.layer_26 import build_W26, build_B26
from layers.layer_27 import build_W27#, build_B26
from layers.layer_28 import build_W28#, build_B26
from layers.layer_29 import build_W29, build_B29
from layers.layer_30_b import build_W30, build_B30
from layers.layer_31_b import build_W31, build_B31
from layers.layer_32_b import build_W32, build_B32
from layers.layer_33_b import build_W33, build_B33
from layers.layer_34 import build_W34, build_B34
from layers.layer_35_c import build_W35, build_B35
from layers.layer_36 import build_W36, build_B36
from layers.layer_37_b import build_W37, build_B37
from layers.layer_38_b import build_W38, build_B38
from layers.layer_39_c import build_W39#, build_B38
from layers.layer_40_b import build_W40, build_B40
from layers.layer_41 import build_W41 #, build_B40

from layers.list_compare import compare_array_lists, get_all_mismatches 
# try:
    # import torch
# except:
    # pass

#Deletion in Graph edit distance

# parameters
eps = 1e-7
C = 1e5 
B = 1e3
# Input
d = 3
m = 10

x =[0.45, 0, 0.59, 0, 0.4, 0.15, 0.11, 0.05, 0.88, 0.55, 0.44, 0.93, 0.52, 0.87, 0.03, 0.33, 0.4, 0, 0.79, 0.65, 0.9]
# one- dimentional label matrix 6 vertices
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

X = x
# X = np.array(X).flatten()
print("Input:")
print("d:", d)
print("m:", m)
print("x:", X)
print("L:", L)
print("A:")
for row in A:
    print(row)
###################################################################################       
#################################################################################my code Graph-uni has it##
# first hidden layer/second layer is L1=W1*X+B1

start_total = time.perf_counter()  # Start timer
start = time.perf_counter()  # Start timer
# first hidden layer/second layer is L1=W1*X+B1
W1 = build_W1(n, d, m, eps)
# Bias matrix for second layer or first Hidden layer 1 ###########
B1 = build_B1(n, d, m, eps)
L1 = np.maximum(W1 @ X + B1, 0)
# print(f"L1 shape: {L1.shape}")  # If numpy array
########################################################################
# To construct the weight matrix for third layer (second hidden layer L2=W2*L1+B2)
W2 = build_W2(n, d, m)
#print(f"W2 shape: {W2.shape}")  
# print(W2)
# Bias matrix for third layer/second hidden layer
B2 = build_B2(d)  
#print(f"B2 length: {len(B2)}") 
L2 = np.maximum(W2 @ L1 + B2, 0)
# print(f"L2 shape: {L2.shape}")  # If numpy array
############################################
# To construct the weight matrix for fourth layer (third hidden layer L3=W3*L2+B3)
W3 = build_W3(d)
# print('weight matrix for fourth layer /third hidden layer,W3')
# print(W3)
L3 = np.maximum(W3 @ L2, 0)
#####################
# To construct the weight matrix for fifth layer (fourth hidden layer L4=W4*L3+B4)
W4 = build_W4(d, eps)
#print(f"W4 shape: {W4.shape}")
B4 = build_B4(d)
#print(f"B4 length: {len(B4)}") 
L4 = np.maximum(W4 @ L3 + B4, 0)
#print(f"L4 shape: {L4.shape}")
#print(L4)
#print('/////////////// 4th layer done \\\\\\\\\\\\\\\\\\')
##################################
# To construct the weight matrix for sixth layer (fifth hidden layer L5=W5*L4+B5)
W5 = build_W5(d, C)
#print(f"W5 shape: {W5.shape}")
B5 = build_B5(d, C)
#print(f"B5 length: {len(B5)}") 
L5 = np.maximum(W5 @ L4 + B5, 0)
#print(f"L5 shape: {L5.shape}")
# print(W5)
# print(B5)
# print(L5)

#print('/////////////// 5th layer done \\\\\\\\\\\\\\\\\\')
###################################
# To construct the weight matrix for seventh layer (sixth hidden layer L6=W6*L5+B6)
W6 = build_W6(d, C)
#print(f"W6 shape: {W6.shape}")
B6 = build_B6(d)
#print(f"B6 length: {len(B6)}") 
L6 = np.maximum(W6 @ L5 + B6, 0)
#print(f"L6 shape: {L6.shape}")
#print('/////////////// 6th layer done \\\\\\\\\\\\\\\\\\')
#####################  
#To construct the weight matrix for eighth layer (seventh hidden layer L7=W7*L6+B7)
W7 = build_W7(d, eps)
#print(f"W7 shape: {W7.shape}")
B7 = build_B7(d)
#print(f"B7 length: {len(B7)}") 
L7 = np.maximum(W7 @ L6 + B7, 0)
#print(f"L7 shape: {L7.shape}")
#####################
#To construct the weight matrix for ninth layer (eighth hidden layer L8=W8*L7+B8)
W8 = build_W8(d)
#print(f"W8 shape: {W8.shape}")
B8 = build_B8(d)
#print(f"B8 length: {len(B8)}") 
L8 = np.maximum(W8 @ L7 + B8, 0)
#print(f"L8 shape: {L8.shape}")
###################################
#To construct the weight matrix for tenth layer (ninth hidden layer L9=W9*L8+B9)
W9 = build_W9(d)
#print(f"W9 shape: {W9.shape}")
# B9 = build_B9(d)
# print(f"B9 length: {len(B9)}") 
L9 = np.maximum(W9 @ L8, 0)
#print(f"L9 shape: {L9.shape}")
#To construct the weight matrix for eleventh layer (tenth hidden layer L10=W10*L9+B10)
W10 = build_W10(d, eps)
#print(f"W10 shape: {W10.shape}")
B10 = build_B10(d, eps)
#print(f"B10 length: {len(B10)}") 
L10 = np.maximum(W10 @ L9 + B10, 0)
#print(f"L10 shape: {L10.shape}")
#To construct the weight matrix for twelfth layer (eleventh hidden layer L11=W11*L10+B11)
W11 = build_W11(d, C)
#print(f"W11 shape: {W11.shape}")
B11 = build_B11(d)
#print(f"B11 length: {len(B11)}") 
L11 = np.maximum(W11 @ L10 + B11, 0)
#print(f"L11 shape: {L11.shape}")
#print('/////////////// 11th layer done \\\\\\\\\\\\\\\\\\')
###################################
W12 = build_W12(d)
#print(f"W11 shape: {W11.shape}")
L12 = np.maximum(W12 @ L11, 0)
#print(f"L12 shape: {L12.shape}")
#print('/////////////// 12th layer done \\\\\\\\\\\\\\\\\\')
###################################################################################       
###################################################################################
#                    Padding 2d number of B at the end of L and 
#                       2d number of rows and columns in A
###################################################################################
###################################################################################
# Create U by padding 2d instances of B to L
U = L + [B] * ( 2*d)
# print("Padded list U:", U)
#
A = np.array(A)
n = A.shape[0]
padded_size = n + 2 * d

# Create a matrix padded with B
V = np.full((padded_size, padded_size), B)

# Place the original A in the top-left corner
V[:n, :n] = A

#### Flatten 2D list to 1D
V1 = [item for row in V for item in row]
# print(V1)
#####################################################################
# To construct weight matrix for fourteenth layer/thirteenth hidden layer is L13=W13*L12+B13
W13 = build_W13(n, d, eps)
#print(f"W13 shape: {W13.shape}")
B13 = build_B13(n, d, eps)
#print(f"B13 length: {len(B13)}") 
L13 = np.maximum(W13 @ L12 + B13, 0)
#print(f"L13 shape: {L13.shape}")
#print('/////////////// 13th layer done \\\\\\\\\\\\\\\\\\')
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
#####################
W14_p = build_W14(n, d, C)
# rows = len(W14)
B14_p = build_B14(n, d, C, B, U) 
#print(f"B14_p length: {len(B14_p)}")
B14_p = B14_p.flatten()
#print(f"B14_p shape after flatten: {B14_p.shape}")
L13_array = np.array(L13).flatten()
#
temp = W14_p @ L13_array
#print(f"W14_p @ L13_array shape: {temp.shape}")

L14_p = np.maximum(temp + B14_p, 0)
#print(f"L14_p shape: {L14_p.shape}")
# Convert 2D array to list of 1D arrays
L14_p_list = [np.array([x]) for x in L14_p]
#
#####################
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
#####################
# To construct weight matrix for Sixteenth layer/fifteenth hidden layer is L15=W15*L14+B15
W15 = []
# eq gi of subs
H15=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        w = 0
        if i == l:
          w = 1
        temp_row.append(w)
    H15.append(temp_row)
##
H16=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        for j in range (1, d+1):
          w = 0
          temp_row.append(w)
    H16.append(temp_row)    
##
H17=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):# w1
        w = 0
        temp_row.append(w)
    H17.append(temp_row)
##
H18=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):#w2
          w = 0
          temp_row.append(w)
    H18.append(temp_row)
##
H19=[]
for l in range(1, n+2*d+1):
    temp_row = []
    for i in range(3*d+1, 7*d+1):
              w = 0
              temp_row.append(w)
    H19.append(temp_row)
for i in range(len(H15)):
    concatenated_row1 = H15[i]+H16[i]+H17[i]+H18[i]+H19[i]
    W15.append(concatenated_row1)
##########

H15=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        w = 0
        temp_row.append(w)
    H15.append(temp_row)
##
H16=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        for j in range(1, d+1):
          w = 0
          if i==l:
              w=1
          temp_row.append(w)
    H16.append(temp_row)    
##
H17=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):# w1
        w = 0
        temp_row.append(w)
    H17.append(temp_row)
##
H18=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):#w2
          w = 0
          temp_row.append(w)
    H18.append(temp_row)
##
H19=[]
for l in range(1, n+2*d+1):
    temp_row = []
    for i in range(3*d+1, 7*d+1):
            w = 0
            temp_row.append(w)
    H19.append(temp_row)    
for i in range(len(H15)):
    concatenated_row2 = H15[i]+H16[i]+H17[i]+H18[i]+H19[i]
    W15.append(concatenated_row2)
##########
# insertion portion
# calculate w'j
##########
H15=[]
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        w = 0
        temp_row.append(w)
    H15.append(temp_row)
##
H16=[]
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        for j in range(1, d+1):
          w = 0
          temp_row.append(w)
    H16.append(temp_row)    
##
H17=[]
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):# w1
        w = 0
        if i == l:
            w=1
        temp_row.append(w)
    H17.append(temp_row)
##
H18=[]
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):#w2
          w = 0
          if i == l:
              w=1
          temp_row.append(w)
    H18.append(temp_row)
##
H19=[]
for l in range(1, d+1):
    temp_row = []
    for i in range(3*d+1, 7*d+1):
            w = 0
            temp_row.append(w)
    H19.append(temp_row)
##    
for i in range(len(H15)):
    concatenated_row3 = H15[i]+H16[i]+H17[i]+H18[i]+H19[i]
    W15.append(concatenated_row3)
##########
# xj
##########
H15=[]
for i in range(3*d+1, 5*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
       w = 0
       temp_row.append(w)
    H15.append(temp_row)
##
H16=[]
for i in range(3*d+1, 5*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
       for j in range(1, d+1):
         w = 0
         temp_row.append(w)
    H16.append(temp_row)    
##
H17=[]
for i in range(3*d+1, 5*d+1):
    temp_row = []
    for l in range(1, d+1):# w1
       w = 0
       temp_row.append(w)
    H17.append(temp_row)
##
H18=[]
for i in range(3*d+1, 5*d+1):
    temp_row = []
    for l in range(1, d+1):#w2
         w = 0
         temp_row.append(w)
    H18.append(temp_row)
##
H19=[]
for i in range(3*d+1, 5*d+1):
        temp_row = []
        for l in range(3*d+1, 7*d+1):
                 w = 0
                 if i==l:
                     w=1
                 temp_row.append(w)
        H19.append(temp_row)    
for i in range(len(H15)):
    concatenated_row4 = H15[i]+H16[i]+H17[i]+H18[i]+H19[i]
    W15.append(concatenated_row4)
# ##########
# # xj for deletion
H15=[]
for i in range(5*d+1, 7*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
       w = 0
       temp_row.append(w)
    H15.append(temp_row)
##
H16=[]
for i in range(5*d+1, 7*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
       for j in range(1, d+1):
         w = 0
         temp_row.append(w)
    H16.append(temp_row)    
##
H17=[]
for i in range(5*d+1, 7*d+1):
    temp_row = []
    for l in range(1, d+1):# w1
       w = 0
       temp_row.append(w)
    H17.append(temp_row)
##
H18=[]
for i in range(5*d+1, 7*d+1):
    temp_row = []
    for l in range(1, d+1):#w2
         w = 0
         temp_row.append(w)
    H18.append(temp_row)
##
H19=[]
for i in range(5*d+1, 7*d+1):
        temp_row = []
        for l in range(3*d+1, 7*d+1):
                 w = 0
                 if i==l:
                     w=1
                 temp_row.append(w)
        H19.append(temp_row)    
for i in range(len(H15)):
    concatenated_row4 = H15[i]+H16[i]+H17[i]+H18[i]+H19[i]
    W15.append(concatenated_row4)
# ######################
# print("weight matrix for Sixteenth layer/fifteenth hidden layer")
# print(W15)
######################
#Bias matrix for Sixteenth layer/fifteenth hidden layer

#B15 = [] is a zero matrix
##################################
L15 = []  #
for i in range(len(W15)):
    temp_row = []
    L15_i_entry = np.maximum((np.dot(W15[i], L14_p_list)), 0)
    L15.append(L15_i_entry)
##################################
############################################
# Remove variables from Python memory
del H15,H16,H17,H18,H19

# Clean RAM
gc.collect()

# Clean GPU memory if PyTorch is available
#try:
    #torch.cuda.empty_cache()
##except:
    ##pass
###########################
del W15        # remove from Python
gc.collect()  # clean RAM

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
###################################
# To construct weight matrix for Seventeenth layer/Sixteenth hidden layer is L16=W16*L15+B16
W16 = []
# output layer of subs
I1=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        w = 0
        if l==i:
          w = 1
        temp_row.append(w)
    I1.append(temp_row)
##
I2=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):
        w = 0
        if l==i:
          w = 1
        temp_row.append(w)
    I2.append(temp_row) 
I3=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):#wj
        w = 0
        temp_row.append(w)
    I3.append(temp_row)
##
I4=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(3*d+1, 5*d+1):
        w = 0
        temp_row.append(w)
    I4.append(temp_row)
###
I5=[]
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(5*d+1, 7*d+1):#xj
        w = 0
        temp_row.append(w)
    I5.append(temp_row)      
for i in range(len(I1)):
    concatenated_row1 = I1[i] + I2[i] + I3[i] + I4[i]+ I5[i]
    W16.append(concatenated_row1)
# print(W16)
######################

# eta1, eta2 for d'= \sum_{j=1}^{d} \delta(x_j, x_{j+d})
I1=[]
for j in range(1, d+1):
    for q in range(2):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I1.append(temp_row)
##
I2=[]
for j in range(1, d+1):
    for q in range(2):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I2.append(temp_row) 
#####
I3=[]
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
      I3.append(temp_row)
##
I4=[]
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*d+1):#deletion input
             w = 0
             temp_row.append(w)
        I4.append(temp_row)
###     
for i in range(len(I1)):
    concatenated_row = I1[i] + I2[i] + I3[i] + I4[i]
    W16.append(concatenated_row)
######################
# eta3, eta4 for d'= \sum_{j=1}^{d} \delta(x_j, x_{j+d})
I1=[]
for j in range(1, d+1):
    for q in range(2):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I1.append(temp_row)
##
I2=[]
for j in range(1, d+1):
    for q in range(2):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I2.append(temp_row) 
#####
I3=[]
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
      I3.append(temp_row)
##
I4=[]
for j in range(1, d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*d+1):#deletion input
             w = 0
             temp_row.append(w)
        I4.append(temp_row)
###     
for i in range(len(I1)):
    concatenated_row2 = I1[i] + I2[i] + I3[i] + I4[i]
    W16.append(concatenated_row2)
######################
#xj as identity map
I1=[]
for j in range(1, 3*d+1):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I1.append(temp_row)
##
I2=[]
for j in range(1, 3*d+1):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I2.append(temp_row) 
#####
I3=[]
for j in range(1, 3*d+1): 
      temp_row = []
      for k in range(1, 3*d+1):
          w = 0
          if j == k:
                w = 1
          temp_row.append(w)
      I3.append(temp_row)
##
I4=[]
for j in range(1, 3*d+1):
        temp_row = []
        for l in range(1, 2*d+1):#deletion input
             w = 0
             temp_row.append(w)
        I4.append(temp_row)
###     
for i in range(len(I1)):
    concatenated_row4 = I1[i] + I2[i] + I3[i] + I4[i]
    W16.append(concatenated_row4)
######################
#xj deletion ionput
I1=[]
for j in range(1, 2*d+1):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I1.append(temp_row)
##
I2=[]
for j in range(1, 2*d+1):
       temp_row = []
       for l in range(1, n+2*d+1):
            w = 0
            temp_row.append(w)
       I2.append(temp_row) 
#####
I3=[]
for j in range(1, 2*d+1): 
      temp_row = []
      for k in range(1, 3*d+1):
          w = 0
          temp_row.append(w)
      I3.append(temp_row)
##
I4=[]
for j in range(1, 2*d+1):
        temp_row = []
        for l in range(1, 2*d+1):#deletion input
             w = 0
             if j == l:
                   w = 1
             temp_row.append(w)
        I4.append(temp_row)
###     
for i in range(len(I1)):
    concatenated_row4 = I1[i] + I2[i] + I3[i] + I4[i]
    W16.append(concatenated_row4)
######################
# print("weight matrix for seventeenth layer/sixteenth hidden layer")
# print(W16)
#####################
#####################
# #Bias matrix for seventeenth layer/sixteenth hidden layer

B16 = []
# bias matrix for subs
for i in range(1, n+2*d+1):
        temp_row = []
        for k in range(1):
          b = 0
          temp_row.append(b)
        B16.append(temp_row)
# bias matrix for eta1, eta2 to solve delta(x_j, x_{j+d})

for j in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=1
          temp_row.append(b)
      B16.append(temp_row)

# bias matrix for eta3, eta4 to solve delta(x_j, x_{j+d})

for j in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
          b = 0
          if q==0:
              b=1
          temp_row.append(b)
      B16.append(temp_row)
# bias matrix for xj
for k in range(1, 5*d+1):
        temp_row = []
        for i in range(1):
          b = 0
          temp_row.append(b)
        B16.append(temp_row) 
# print('Printing B16')
# for i in B16:
#     print(i)
##################################
L16 = []  # output layer of substitution
for i in range(len(W16)):
    temp_row = []
    L16_i_entry = np.maximum((np.dot(W16[i], L15)+B16[i]), 0)
    L16.append(L16_i_entry)
############################
#print('Printing output layer of substitution for sixteenth layer')
# for i in L16:
#     print(i)

##################################

############################
# Remove variables from Python memory
del I1,I2, I3,I4, I5

# Clean RAM
gc.collect()

###########################
del W16,B16        # remove from Python
gc.collect()  # clean RAM

#####################
del L15        # remove from Python
gc.collect()  # clean RAM

#try:
    #torch.cuda.empty_cache()  # clean GPU memory
##except:
    ##pass
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for eighteenth layer/seventeenth hidden layer is L17=W17*L16+B17
W17 = []
#subs nodes as identity map
I6 = []
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
for i in range(1, n+2*d+1):
    temp_row = []
    for j in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    I16.append(temp_row)
I7 = []
for i in range(1, n+2*d+1):
    temp_row = []
    for j in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    I7.append(temp_row)
##########
I8 = []
for i in range(1, n+2*d+1):
    temp_row = []
    for j in range(1, 5*d+1):
           w = 0
           temp_row.append(w)
    I8.append(temp_row)
##########
#############    
for i in range(len(I6)):
    concatenated_row1 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row1)
########################
# tau2 for g1_j that corresponding \max \Big( x_{j+2d} - C(1-\delta(x_j, x_{j+d})), 0 \Big)
I6 = []
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =C
                else:
                    w=-C
            temp_row.append(w)
        I16.append(temp_row)
I7 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =C
                else:
                    w=-C
            temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 5*d+1):
            w = 0
            if i+2*d==j:
                w=1
            temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row2 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row2)
########################
# tau3 for g2_j that corresponding \max \Big( B - C \, \delta(x_j, x_{j+d}), 0 \Big)
I6 = []
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =-C
                else:
                    w=C
            temp_row.append(w)
        I16.append(temp_row)
I7 = []
for j in range(1, d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            if j == k:
                if q==0:
                  w =-C
                else:
                    w=C
            temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 5*d+1):
            w = 0
            temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row3 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row3)
########################
# for eta5 that corresponding to delta(xj,x{j+d})
I6 = []
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
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
        I16.append(temp_row)
I7 = []
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
        I7.append(temp_row)
##########
I8 = []
for i in range(1, d+1):
    temp_row = []
    for j in range(1, 5*d+1):
           w = 0
           temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row7 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row7)
########################
# xj for insertion
I6 = []
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        I16.append(temp_row)
I7 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, 2*d+1):
    temp_row = []
    for j in range(1, 5*d+1):
           w = 0
           if i==j:
               w=1
           temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row5 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row5)
########################
# xj for deletion
I6 = []
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta1, eta2 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        I16.append(temp_row)
I7 = []
for j in range(1, 2*d+1):
        temp_row = []
        for k in range(1, d+1):# eta3, eta4 nodes
          for q in range(2):
            w = 0
            temp_row.append(w)
        I7.append(temp_row)
##########
I8 = []
for i in range(1, 2*d+1):
    temp_row = []
    for j in range(1, 5*d+1):
           w = 0
           if i+3*d==j:
               w=1
           temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row6 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row6)
########################
# d'=summation delta(xj,x{j+d})
I6 = []
for i in range(1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs
        w = 0
        temp_row.append(w)
    I6.append(temp_row)
##########
I16 = []
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
        I16.append(temp_row)
I7 = []
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
        I7.append(temp_row)
##########
I8 = []
for i in range(1):
    temp_row = []
    for j in range(1, 5*d+1):
           w = 0
           temp_row.append(w)
    I8.append(temp_row)
##########   
for i in range(len(I6)):
    concatenated_row7 = I6[i] +I16[i] + I7[i] + I8[i] 
    W17.append(concatenated_row7)
########################
######################
# print("weight matrix for eighteenth layer/seventeenth hidden layer")
# print(W17)
#####################
# #Bias matrix for eighteenth layer/seventeenth hidden layer

B17 = []
# bias matrix subs
for i in range(1, n+2*d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B17.append(temp_row)

# bias matrix for tau2
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -2*C
          temp_row.append(b)
        B17.append(temp_row) 
# bias matrix for tau3
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = B+C
          temp_row.append(b)
        B17.append(temp_row)
# bias matrix for eta5
for i in range(1, d+1):
        temp_row = []
        for k in range(1):
          b = -1
          temp_row.append(b)
        B17.append(temp_row)
# bias matrix for insertion
for l in range(1, 2*d+1):
        temp_row = []
        for k in range(1):
            w = 0
            temp_row.append(w)
        B17.append(temp_row) 
# bias matrix for deletion
for l in range(1, 2*d+1):
        temp_row = []
        for k in range(1):
            w = 0
            temp_row.append(w)
        B17.append(temp_row) 
# bias for d'
for i in range(1):
      temp_row = []
      for k in range(1):
        b = -d
        temp_row.append(b)
      B17.append(temp_row)        
##################################
# print('Printing B17')
# for i in B17:
#     print(i)
##################################
L17 = []  # d' and eta5 nodes
for i in range(len(W17)):
    temp_row = []
    L17_i_entry = np.maximum((np.dot(W17[i], L16)+B17[i]), 0)
    L17.append(L17_i_entry)
###################################
d1 = L17[-1]  # this gets the last element that is d'
# print(d1)
d2 = int(d1[0])
# print(d2)
###########################
#####################
del L16        # remove from Python
gc.collect()  # clean RAM

############################################
# Remove variables from Python memory
del I6,I16, I7,I8

# Clean RAM
gc.collect()

###########################
del W17, B17        # remove from Python
gc.collect()  # clean RAM

#
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
###################################
# To construct weight matrix for ninteenth layer/eighteenth hidden layer is L18=W18*L17+B18
W18 = []
I16 = []  # subs nodes as identity map
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  # subs nodes as identity map
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  # subs nodes as identity map
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
##########
for i in range(len(I16)):
    concatenated_row = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row)
# print(W18)
######################

## calculate tau4 for gj, tau4=tau2+tau3 
I16 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1,d+1):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
##########
for i in range(len(I16)):
    concatenated_row2 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row2)
# print(W18)
######################
## eta5 as identity map
I16 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, d+1):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1,d+1):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
##########
for i in range(len(I16)):
    concatenated_row3 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row3)
# print(W18)
######################
## # eta19, eta20 nodes to calculate H(-n-d'-1+e'_j)
I16 = []  
for i in range(1, d+1):
  for q in range(2):  
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, d+1):
  for q in range(2):   
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, d+1):
  for q in range(2):    
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, d+1):
   for q in range(2):   
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for i in range(1, d+1):
   for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  #insertion
        w = 0
        if i == l:
            w = 1/eps
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1,d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
##########
for i in range(len(I16)):
    concatenated_row4 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row4)
# print(W18)
######################
# # eta7,eta8 nodes to find H(x{j+d}-n-d'-1)
I16 = []  
for i in range(1, d+1):
  for q in range(2):    
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        if k+d==j:
            w=1/eps
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1,d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
##########
for i in range(len(I16)):
    concatenated_row5 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row5)
# print(W18)
######################
## eta 9, eta 10 for H(n-i)
I16 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row6 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row6)
# print(W18)
######################
# eta 11, eta 12 for H(i-n-1)
I16 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row7 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row7)
# print(W18)
######################
# eta 13, eta 14 for H(n+d'-i)
I16 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row8 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row8)
# print(W18)
######################
# eta 15, eta 16 for H(n+d'-k)
I16 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row9 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row9)
# print(W18)
######################
# eta 17, eta 18 for H(k-n-1)
I16 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, n+2*d+1):
  for q in range(2):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row10 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row10)
# print(W18)
######################
# xj for insertion as identity map
I16 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, 2*d+1):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        if k==j:
            w=1
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row11 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row11)
# print(W18)
######################
# xj for deletion as identity map
I16 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, n+2*d+1):  # subs nodes
        w = 0
        temp_row.append(w)
    I16.append(temp_row)
##########
I17 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1,d+1):  # tau2
        w = 0
        temp_row.append(w)
    I17.append(temp_row)
##########
I18 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, d+1):  # tau3
        w = 0
        temp_row.append(w)
    I18.append(temp_row)
##########
I19 = []  #
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, d+1):  # eta5
        w = 0
        temp_row.append(w)
    I19.append(temp_row)
##########
I20 = []  
for k in range(1, 2*d+1):
    temp_row = []
    for j in range(1, 2*d+1):  #insertion
        w = 0
        temp_row.append(w)
    I20.append(temp_row)
##########
I21 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1, 2*d+1):  # deletion nodes
        w = 0
        if i==l:
            w=1
        temp_row.append(w)
    I21.append(temp_row)
##########
I22 = []  
for i in range(1, 2*d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    I22.append(temp_row)
##########
for i in range(len(I16)):
    concatenated_row12 = I16[i] + I17[i] + I18[i]+ I19[i]+\
        I20[i] + I21[i] + I22[i]
    W18.append(concatenated_row12)
# print(W18)
######################
# #Bias matrix for eighteenth layer

B18 = []
# bias matrix for subs

for i in range(1, n+2*d+1):
    temp_row = []
    for k in range(1):
        w = 0
        temp_row.append(w)
    B18.append(temp_row)

# tau4(gj) nodes 

for l in range(1, d+1):
        temp_row = []
        for k in range(1):
            w = 0
            temp_row.append(w)
        B18.append(temp_row)
        
#eta5 nodes as identity map
for l in range(1, d+1):
        temp_row = []
        for k in range(1):
            w = 0
            temp_row.append(w)
        B18.append(temp_row)  
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
            B18.append(temp_row)
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
        B18.append(temp_row) 
#################
# bias matrix for eta9,eta10
for i in range(1, n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(n-i)/eps
          else:
              b=(n-i)/eps
          temp_row.append(b)
        B18.append(temp_row)         
################## 
# bias matrix for eta11,eta12
for i in range(1, n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(i-n-1)/eps
          else:
              b=(i-n-1)/eps
          temp_row.append(b)
        B18.append(temp_row)         
################## 
# bias matrix for eta13,eta14
for i in range(1, n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          b = 0
          if q==0:
             b=1+(n+d2-i)/eps
          else:
              b=(n+d2-i)/eps
          temp_row.append(b)
        B18.append(temp_row)         
################## 
# bias matrix for eta15,eta16
for k in range(1, n+2*d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
          b = 0
          if q==0:
             b=1+(n+d2-k)/eps
          else:
              b=(n+d2-k)/eps
          temp_row.append(b)
        B18.append(temp_row)         
################## 
# bias matrix for eta17,eta18
for k in range(1, n+2*d+1):
    for q in range(2):
        temp_row = []
        for i in range(1):
          b = 0
          if q==0:
             b=1+(k-n-1)/eps
          else:
              b=(k-n-1)/eps
          temp_row.append(b)
        B18.append(temp_row)     
##################            
# bias for xj nodes
for j in range(1, 2*d+1):   
          temp_row = []
          for l in range(1):
              b = 0
              temp_row.append(b)
          B18.append(temp_row)
# bias for xj nodes
for j in range(1, 2*d+1):   
          temp_row = []
          for l in range(1):
              b = 0
              temp_row.append(b)
          B18.append(temp_row)          
##################################
# print('Printing B18')
# for i in B18:
#     print(i)
##################################
L18 = []  # 
for i in range(len(W18)):
    temp_row = []
    L18_i_entry = np.maximum((np.dot(W18[i], L17)+B18[i]), 0)
    L18.append(L18_i_entry)
##################################
# Remove variables from Python memory
del I16,I17,I18, I19,I20,I21,I22

# Clean RAM
gc.collect()

###########################
del W18,B18        # remove from Python
import gc
gc.collect()  # clean RAM
W19_p = build_W19(n, d, eps)
#rows = len(W19)
B19_p = build_B19(d, n) 
#print(f"B19_p length: {len(B19_p)}")
B19_p = B19_p.flatten()
#print(f"B19_p shape after flatten: {B19_p.shape}")
L18_array = np.array(L18).flatten()  # Changed from just np.array(L18)
#print(f"L18 length: {len(L18)}")
temp = W19_p @ L18_array
#print(f"W19_p @ L18_array shape: {temp.shape}")  # Should be (260,)

L19_p = np.maximum(temp + B19_p, 0)
#print(f"L19_p shape: {L19_p.shape}")  # Should be (260,) now
L19_p_list = [np.array([x]) for x in L19_p]
# 
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
# To construct weight matrix for twentieth layer is L20=W20*L19+B20
W20_p = build_W20(n, d, C)
# rows = len(W20)
B20_p = build_B20(n, d, B, C)
# #print(f"B20_p length: {len(B20_p)}")
##B20_p = B20_p.flatten()  # Convert sparse → dense
B20_p = B20_p.toarray().flatten()
## print(f"B20_p shape after flatten: {B20_p.shape}")
L19_array = np.array(L19_p_list).flatten()  # Changed from just np.array(L20)
#print(f"L20 length: {len(L20)}")
temp = W20_p @ L19_array
##temp = W20_p @ L20_p_list

L20_p = np.maximum(temp + B20_p, 0)
##print(f"L20_p shape: {L20_p.shape}")  # Should be (260,) now

L20_p_list = [np.array([x]) for x in L20_p]
#
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
#####################
# # To construct weight matrix for twenty one layer is L21=W21*L20+B21
W21_p = build_W21(n, d, eps)
# rows = len(W21)
B21_p = build_B21(n, d, V1, eps, C)
# #print(f"B21_p length: {len(B21_p)}")
B21_p = B21_p.toarray().flatten()
## print(f"B21_p shape after flatten: {B21_p.shape}")
L20_array = np.array(L20_p_list).flatten()  # Changed from just np.array(L21)
#print(f"L20 length: {len(L20)}")
temp = W21_p @ L20_array
##temp = W21_p @ L20_p_list
#print(f"W21_p @ L20_array shape: {temp.shape}")  # Should be (260,)

L21_p = np.maximum(temp + B21_p, 0)
##print(f"L21_p shape: {L21_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L21_p_list = [np.array([x]) for x in L21_p]

#######################
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty two layer is L22=W22*L21+B22
W22_p = build_W22(n, d, C)
# rows = len(W22)

B22_p = build_B22(n, d, C) 
# #print(f"B22_p length: {len(B22_p)}")
B22_p = B22_p.toarray().flatten()  # Convert sparse → dense
#print(f"B22_p shape after flatten: {B22_p.shape}")
L21_array = np.array(L21_p_list).flatten()  # Changed from just np.array(L22)
# print(f"L36 length: {len(L36)}")
temp = W22_p @ L21_array
# print(f"W22_p @ L36_array shape: {temp.shape}")  # Should be (260,)

L22_p = np.maximum(temp + B22_p, 0)
#print(f"L22_p shape: {L22_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L22_p_list = [np.array([x]) for x in L22_p]
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty third layer is L23=W23*L22+B23
W23_p = build_W23(n, d)
# rows = len(W23)
B23_p = build_B23(n, d)
# #print(f"B23_p length: {len(B23_p)}")
B23_p = B23_p.toarray().flatten()
## print(f"B23_p shape after flatten: {B23_p.shape}")
L22_array = np.array(L22_p_list).flatten()  # Changed from just np.array(L23)
#print(f"L22 length: {len(L22)}")
temp = W23_p @ L22_array

L23_p = np.maximum(temp + B23_p, 0)
##print(f"L23_p shape: {L23_p.shape}")  # Should be (260,) now
L23_p_list = [np.array([x]) for x in L23_p]
#print('!!!!!!!!!!!!!!!!!!!!!! My modification ends: 23rd')
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty four layer is L24=W24*L23+B24 
W24_p = build_W24(n, d, C, eps)
# rows = len(W24)
B24_p = build_B24(n, d, C, eps)
# #print(f"B24_p length: {len(B24_p)}")
B24_p = B24_p.toarray().flatten()
## print(f"B24_p shape after flatten: {B24_p.shape}")
L23_array = np.array(L23_p_list).flatten()  # Changed from just np.array(L24)
#print(f"L24 length: {len(L24)}")
temp = W24_p @ L23_array

L24_p = np.maximum(temp + B24_p, 0)
##print(f"L24_p shape: {L24_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L24_p_list = [np.array([x]) for x in L24_p]
############################################
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty fifth that is second layer of deletion L25=W25*L24+B25
##################################
W25_p = build_W25(n, d)
# rows = len(W25)
B25_p = build_B25(n, d, C) 
# #print(f"B25_p length: {len(B25_p)}")
B25_p = B25_p.toarray().flatten()  # Convert sparse → dense
#print(f"B25_p shape after flatten: {B25_p.shape}")
L24_array = np.array(L24_p_list).flatten()  # Changed from just np.array(L25)
# print(f"L21 length: {len(L21)}")
temp = W25_p @ L24_array
# print(f"W25_p @ L21_array shape: {temp.shape}")  # Should be (260,)

L25_p = np.maximum(temp + B25_p, 0)
#print(f"L25_p shape: {L25_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L25_p_list = [np.array([x]) for x in L25_p]
#
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty sixth layer that is third layer of deletion L26=W26*L25+B26
#######################
#print('Printing nodes of twenty sixth layer that is third layer of deletion')

W26_p = build_W26(n, d, C)
# rows = len(W26)
B26_p = build_B26(n, d, C)
# #print(f"B26_p length: {len(B26_p)}")
##B26_p = B26_p.flatten()  # Convert sparse → dense
B26_p = B26_p.toarray().flatten()
## print(f"B26_p shape after flatten: {B26_p.shape}")

B26_converted = [[int(x)] for x in B26_p]
##print(f"B26~~~~~~~~~~~~~~~~: {B26}")
L25_array = np.array(L25_p_list).flatten()  # Changed from just np.array(L26)
#print(f"L25 length: {len(L25)}")
temp = W26_p @ L25_array
##temp = W26_p @ L25_p_list
#print(f"W26_p @ L25_array shape: {temp.shape}")  # Should be (260,)

L26_p = np.maximum(temp + B26_p, 0)
##print(f"L26_p shape: {L26_p.shape}")  # Should be (260,) now
# Convert 2D array to list of 1D arrays
L26_p_list = [np.array([x]) for x in L26_p]
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty seventh layer that is Output layer if insertion and fourth layer of deletion L27=W27*L26+B27
W27_p = build_W27(n, d)
# rows = len(W27)
L26_array = np.array(L26_p_list).flatten()  # Changed from just np.array(L27)
#print(f"L27 length: {len(L27)}")
temp = W27_p @ L26_array

L27_p = np.maximum(temp, 0)
##print(f"L27_p shape: {L27_p.shape}")  # Should be (260,) now
# Convert 2D array to list of 1D arrays
L27_p_list = [np.array([x]) for x in L27_p]
#
#######################
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for twenty eight layer that is fifth layer of deletion is L28=W28*L27+B28
W28_p = build_W28(n, d)
# rows = len(W28)
L27_array = np.array(L27_p_list).flatten()  # Changed from just np.array(L28)
#print(f"L28 length: {len(L28)}")
temp = W28_p @ L27_array

L28_p = np.maximum(temp, 0)
##print(f"L28_p shape: {L28_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L28_p_list = [np.array([x]) for x in L28_p]
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
#####################
W29_p = build_W29(n, d, eps)
# rows = len(W29)
B29_p = build_B29(n, d)
# #print(f"B29_p length: {len(B29_p)}")
##B29_p = B29_p.flatten()  # Convert sparse → dense
B29_p = B29_p.toarray().flatten()
## print(f"B29_p shape after flatten: {B29_p.shape}")

B29_converted = [[int(x)] for x in B29_p]
#print(f"B29~~~~~~~~~~~~~~~~: {B29}")
# # FIXED: Convert AND flatten L29
L28_array = np.array(L28_p_list).flatten()  # Changed from just np.array(L29)
#print(f"L29 length: {len(L29)}")
temp = W29_p @ L28_array
##temp = W29_p @ L29_p_list
#print(f"W29_p @ L29_array shape: {temp.shape}")  # Should be (260,)

L29_p = np.maximum(temp + B29_p, 0)
##print(f"L29_p shape: {L29_p.shape}")  # Should be (260,) now
# Convert 2D array to list of 1D arrays
L29_p_list = [np.array([x]) for x in L29_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for thirtieth layer is L30=W30*L29+B30
##################################
W30_p = build_W30(n, d)
# rows = len(W30)
B30_p = build_B30(n, d)
# #print(f"B30_p length: {len(B30_p)}")
B30_p = B30_p.toarray().flatten()
## print(f"B30_p shape after flatten: {B30_p.shape}")

L29_array = np.array(L29_p_list).flatten()  # Changed from just np.array(L30)
temp = W30_p @ L29_array
##temp = W30_p @ L30_p_list
#print(f"W30_p @ L30_array shape: {temp.shape}")  # Should be (260,)
L30_p = np.maximum(temp + B30_p, 0)
##print(f"L30_p shape: {L30_p.shape}")  # Should be (260,)
# Convert 2D array to list of 1D arrays
L30_p_list = [np.array([x]) for x in L30_p]
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
W31_p = build_W31(n, d, C)
# rows = len(W31)
B31_p = build_B31(n, d, C)
# #print(f"B31_p length: {len(B31_p)}")
##B31_p = B31_p.flatten()  # Convert sparse → dense
B31_p = B31_p.toarray().flatten()
## print(f"B31_p shape after flatten: {B31_p.shape}")
L30_array = np.array(L30_p_list).flatten()  # Changed from just np.array(L31)
#print(f"L31 length: {len(L31)}")
temp = W31_p @ L30_array
##temp = W31_p @ L31_p_list
L31_p = np.maximum(temp + B31_p, 0)
#print(f"L31_p shape: {L31_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L31_p_list = [np.array([x]) for x in L31_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for thirty two layer is L32=W32*L31+B32
W32_p = build_W32(n, d, eps)
#
B32_p = build_B32(n, d, eps)
# #
B32_p = B32_p.toarray().flatten()
# # FIXED: Convert AND flatten L32
L31_array = np.array(L31_p_list).flatten()  # Changed from just np.array(L32)
#
temp = W32_p @ L31_array
##
L32_p = np.maximum(temp + B32_p, 0)
##print(f"L32_p shape: {L32_p.shape}")  # Should be (260,) now
L32_p_list = [np.array([x]) for x in L32_p]
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for eleventh layer/tenth hidden layer is L10=W10*L9+B10
W33_p = build_W33(n, d)
B33_p = build_B33(n, d, C)
# #print(f"B33_p length: {len(B33_p)}")
##B33_p = B33_p.flatten()  # Convert sparse → dense
B33_p = B33_p.toarray().flatten()
## print(f"B33_p shape after flatten: {B33_p.shape}")
L32_array = np.array(L32_p_list).flatten()  # Changed from just np.array(L33)
#print(f"L32 length: {len(L32)}")
temp = W33_p @ L32_array
##
L33_p = np.maximum(temp + B33_p, 0)
##print(f"L33_p shape: {L33_p.shape}")  # Should be (260,) now
L33_p_list = [np.array([x]) for x in L33_p]
#
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for thirty four layer is L34=W34*L33+B34
W34_p = build_W34(n, d)
# rows = len(W34)
B34_p = build_B34(n, d)
# 
B34_p = B34_p.toarray().flatten()
##
L33_array = np.array(L33_p_list).flatten()  # Changed from just np.array(L34)

temp = W34_p @ L33_array
L34_p = np.maximum(temp + B34_p, 0)
##print(f"L34_p shape: {L34_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L34_p_list = [np.array([x]) for x in L34_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for thirty five layer is L35=W35*L34+B35
###################################
W35_p = build_W35(n, d, eps)
# rows = len(W35)

B35_p = build_B35(n, d)
# #print(f"B35_p length: {len(B35_p)}")

B35_p = B35_p.toarray().flatten()

L34_array = np.array(L34_p_list).flatten()  # Changed from just np.array(L35)

temp = W35_p @ L34_array

L35_p = np.maximum(temp + B35_p, 0)
##print(f"L35_p shape: {L35_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L35_p_list = [np.array([x]) for x in L35_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for thirty sixth layer is L36=W36*L35+B36

###################################

W36_p = build_W36(n, d, B, C)

B36_p = build_B36(n, d, C) 
# #print(f"B36_p length: {len(B36_p)}")
B36_p = B36_p.toarray().flatten()  # Convert sparse → dense

L35_array = np.array(L35_p_list).flatten()  # Changed from just np.array(L36)
#
temp = W36_p @ L35_array
# print(f"W36_p @ L35_array shape: {temp.shape}")  # Should be (260,)

L36_p = np.maximum(temp + B36_p, 0)
#print(f"L36_p shape: {L36_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L36_p_list = [np.array([x]) for x in L36_p]
#

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################

W37_p = build_W37(n, d, eps)

B37_p = build_B37(n, d , eps, B) 

L36_array = np.array(L36_p_list).flatten()  # Changed from just np.array(L37)
#print(f"L36 length: {len(L36)}")
temp = W37_p @ L36_array
##

L37_p = np.maximum(temp + B37_p, 0)
#print(f"L37_p shape: {L37_p.shape}")  # Should be (260,) now
# Convert 2D array to list of 1D arrays
L37_p_list = [np.array([x]) for x in L37_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for Thirty eight layer is L38=W38*L37+B38
#################################

W38_p = build_W38(n, d, C)
#rows = len(W38)
B38_p = build_B38(n, d , C) 
# #print(f"B38_p length: {len(B38_p)}")
B38_p = B38_p.flatten()

L37_array = np.array(L37_p_list).flatten()  # Changed from just np.array(L38)

temp = W38_p @ L37_array
#print(f"W38_p @ L37_array shape: {temp.shape}")  # Should be (260,)

L38_p = np.maximum(temp + B38_p, 0)
#print(f"L38_p shape: {L38_p.shape}")  # Should be (260,) now
# Convert 2D array to list of 1D arrays
L38_p_list = [np.array([x]) for x in L38_p]

############################################
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
# To construct weight matrix for thirty ninth layer is L39=W39*L38+B39
############################

W39_p = build_W39(n, d, eps)

L38_array = np.array(L38_p_list).flatten()  

L39_p = np.maximum(W39_p @ L38_array, 0)
#print(f"L39_p shape: {L39_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L39_p_list = [np.array([x]) for x in L39_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for fortieth layer is L40=W40*L39+B40
###################################
#print('Printing zeta5 nodes for fortieth layer')

#print('!!!!!!!!!!!!!!!!!!!!!! Modified 40th layer') ###############)

W40_p = build_W40(n, d, C)
# rows = len(W40)

B40_p = build_B40(n, d, C)
# #print(f"B40_p length: {len(B40_p)}")
##B40_p = B40_p.flatten()  # Convert sparse → dense
B40_p = B40_p.toarray().flatten()

L39_array = np.array(L39_p_list).flatten()  # Changed from just np.array(L40)

temp = W40_p @ L39_array


L40_p = np.maximum(temp + B40_p, 0)
##print(f"L40_p shape: {L40_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
L40_p_list = [np.array([x]) for x in L40_p]

end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")
start = time.perf_counter()  # Start timer
############################################
# To construct weight matrix for forty one layer is L41=W41*L40+B41

W41_p = build_W41(n, d)
L40_array = np.array(L40_p_list).flatten()  # Changed from just np.array(L41)

temp = W41_p @ L40_array
##temp = W41_p @ L41_p_list
#print(f"W41_p @ L40_array shape: {temp.shape}")  # Should be (260,)

Y1_p = np.maximum(temp, 0)
##print(f"L41_p shape: {L41_p.shape}")  # Should be (260,) now

# Convert 2D array to list of 1D arrays
Y1_p_list = Y1_p.tolist()

###########
end = time.perf_counter()    # Stop timer
time_taken = end - start
#print(f"~~~~~~~~~~~~~~~~~~~Time: {time_taken:.4f} seconds")

limit = n + d
#print(f"limit: {limit}")
# Convert your list to NumPy array and flatten
Y1_p_list = np.array(Y1_p).flatten()

# Separate the parts
first_part = Y1_p_list[:limit]
rest_part = Y1_p_list[limit:]

# Remove zeros only from the first part
first_part_no_zeros = first_part[first_part != 0]

# Concatenate back
Y = np.concatenate([first_part_no_zeros, rest_part])
Y = np.array(Y).flatten()
#print(Y)
#########################################
# Flatten the output
# flat_Y = [y[0] for y in Y]
flat_Y = Y

# Extract L' (first n values)
L_star = flat_Y[:n+d]

# Extract A' (remaining n*n values and reshape to n x n matrix)
A_star_flat = flat_Y[n+d:]
A_star = [A_star_flat[i * (n+d) : (i + 1) * (n+d)] for i in range(n+d)]
#####################################  
# Remove B and convert remaining to integers
L_filtered = [int(x) for x in L_star if x != B and x!=0]

print("L' =", L_filtered)
#########################################
# Keep rows that are NOT fully padding
keep_rows = [i for i in range(n+d) if not all(val == B for val in A_star[i])]

# Keep columns that are NOT fully padding
keep_cols = [
    j for j in range(n+d)
    if not all(A_star[i][j] == B for i in range(n+d))
]

# Build the reduced adjacency matrix
A_reduced = [
    [int(A_star[i][j]) for j in keep_cols]
    for i in keep_rows
]

print("A' =")
for row in A_reduced:
    print(row)
#####################################