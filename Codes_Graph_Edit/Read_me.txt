File: GS_d.py
This file contains an implementation of the GS_d-generative ReLU to generate graphs within a given edit distance due to substitution.

Input:
- L: Label matrix of given graph 
- A: Adjacency matrix of given graph 
- d: Edit distance
- m: Size of the symbol set
- x: String of length 2d to identify substitution operations

Output:
- L': Label matrix of output graph
- A': Adjacency matrix of output graph

Example:
L = [3, 5, 4, 2, 4]
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
d = 3
m = 5
x =  [5, 3, 3, 5, 2, 3]
L' = [3, 5, 2, 2, 5]
A' = [0, 1, 0, 0, 1]
     [1, 0, 1, 1, 1]
     [0, 1, 0, 0, 0]
     [0, 1, 0, 0, 1]
     [1, 1, 0, 1, 0]

--------------------------------------------------

File: GD_d.py
This file contains an implementation of the GD_d-generative ReLU to generate graphs within a given edit distance due to deletion only.

Input:
- L: Label matrix of given graph 
- A: Adjacency matrix of given graph 
- d: Edit distance
- m: Size of the symbol set
- x: String of length 2d to identify deletion operations

Output:
- L': Label matrix of output graph
- A': Adjacency matrix of output graph

Example:
L = [3, 5, 4, 2, 4]
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    ]
d = 3
m = 5
x =  [5, 3, 3, 5, 2, 3]
L' = [3, 5, 2, 4]
A' = [0, 1, 0, 1]
     [1, 0, 1, 1]
     [0, 1, 0, 1]
     [1, 1, 1, 0]

--------------------------------------------------

File: GI_d.py
This file contains an implementation of the GI_d-generative ReLU to generate graphs within a given edit distance due to insertion only.

Input:
- L: Label matrix of given graph 
- A: Adjacency matrix of given graph 
- d: Edit distance
- m: Size of the symbol set
- x: String of length 3d to identify insertion operations

Output:
- L': Label matrix of output graph
- A': Adjacency matrix of output graph

Example:
L = [3, 5, 4, 2, 4]
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
d = 3
m = 5
x = [4, 3, 7, 6, 3, 2, 5, 1, 2]
L' = [3, 5, 4, 2, 4, 5]
A' = [0, 1, 0, 0, 1, 0]
     [1, 0, 1, 1, 1, 0]
     [0, 1, 0, 0, 0, 0]
     [0, 1, 0, 0, 1, 1]
     [1, 1, 0, 1, 0, 0]
     [0, 0, 0, 1, 0, 0]

--------------------------------------------------

File: GE_d_unified.py
This file contains an implementation of the GE_d-generative ReLU to generate graphs within a given edit distance due to substitution, deletion, and insertion operations simultaneously.

Input:
- L: Label matrix of given graph 
- A: Adjacency matrix of given graph 
- d: Edit distance
- m: Size of the symbol set
- x: String of length 7d to identify substitution, deletion, and insertion operations

Output:
- L': Label matrix of output graph
- A': Adjacency matrix of output graph

Example:
L = [3, 5, 4, 2, 4]
A = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    ]
d = 3
m = 10
x = [0.45, 0, 0.59, 0, 0.4, 0.15, 0.11, 0.05, 0.88, 0.55, 0.44, 0.93, 0.52, 0.87, 0.03, 0.33, 0.4, 0, 0.79, 0.65, 0.9]
L' = [3, 5, 1, 2, 4, 1]
A' = [0, 1, 0, 1, 1, 0]
     [1, 0, 1, 1, 1, 0]
     [0, 1, 0, 0, 0, 0]
     [1, 1, 0, 0, 1, 0]
     [1, 1, 0, 1, 0, 0]
     [0, 0, 0, 0, 0, 0]