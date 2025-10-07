import time
import numpy as np
from graph_fact_tools import *
import pickle

#Below we have the graphs to be factored
A = np.array([
    [1,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,1],
], dtype=int) # Our 4-star

A1 = np.array([
    [1,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,0],
    [0,0,1,1,0,0,0,0], #A special splitting of the degree 4 vertex happens here
    [0,0,0,0,1,1,0,0], #It splits the row .. 0 0 1 1 1 1 0 0 .. to
    [0,0,0,0,1,0,1,0], # .. 0 0 1 1 0 0 0 0 ..
    [0,0,0,0,0,1,0,1], # .. 0 0 0 0 1 1 0 0 ..
], dtype=int)

#E is the graph that implements this splitting
E = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
], dtype=int)

input = [[A],[E,A1]] # [E,A1] covers a special splitting of the degree 4 vertex not considered by our algorithm

print("Starting... \n")
t0 = time.perf_counter()
result = factorize(input)
t1 = time.perf_counter()
print(f"Time elapse: {t1-t0:.6f} s") # It takes me about 2 hours to run this.

# We verify that the product in every list is still the same
prod_val = all([np.array_equal(np.linalg.multi_dot(list),A) for list in result])

with open("result.pkl", "wb") as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

# To load later:
# with open("result.pkl", "rb") as f:
#     result = pickle.load(f)

print(f"Matrix lists retain total product: {prod_val}")

print(len(result), "lists produced")
