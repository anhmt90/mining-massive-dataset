#!/usr/bin/env python
# coding: utf-8

# # Project task 04:  Restaurant ranking

# In[1]:


import numpy as np
import scipy.sparse as sp


# The goal of this task is to rank restaurants using the **PageRank** algorithm. You are given a directed weighted graph where each node represents one restaurant. The edges in this graph are based on users reviews.
# 
# Additionally for each restaurant you are given the categories it belongs to, i.e. 'Mexican', 'Italian', etc. Note that each restaurant can belong to multiple categories.
# 
# Considering these categories as topics you will perform **topic-sensitive PageRank**, enabling you to e.g. find the top 10 'Mexican' restaurants.

# ## 1. Load data
# 
# * The graph is stored as a sparse adjacency matrix $A$
# * The categories are stored in a binary sparse matrix $C$, with $C_{ij}=1$ indicating that restaurant $i$ belongs to category $j$
# * We also provide you with a dictionary mapping each category to its corresponding column index in $C$
# * The name of each restaurant is provided as a list, with the i-th element in the list corresponding to the i-th node in the graph


A = sp.load_npz('task_04_data/restaurant_graph.npz')


C = sp.load_npz('task_04_data/restaurant_categories.npz')

categories = np.load('task_04_data/categories.npy', allow_pickle = True).tolist()
categories['Mexican'], categories['Chinese']



names = np.load('task_04_data/restaurant_names.npy', allow_pickle = True)
names[:3]



assert A.shape[0] == len(names) == C.shape[0]
assert C.shape[1] == len(categories)


A.shape, C.shape


#  ## 2. Determine the teleport set
#  
# 
# Given a list of topics of intereset, i.e. `['Mexican', 'Italian', ...]`, implement a helper function to return all the restaurants that belong to **at least one** of these topics. These restaurants will become part of the teleport set in topic-sensitive PageRank.

# In[20]:


def teleport_set(C, topics, categories):
    """
    Finds the teleport set consisting of restaurants that belong to at least one of the specified topics.
    
    Parameters
    ----------
    C             : sp.spmatrix, shape [num_restaurants, num_categories]
                    Binary matrix encoding which restaurants belongs to which categories.
    topics        : List[string]
                    List of topics of interest.
    categories    : dict(string, int)
                    Dictionary mapping each category to its corresponding column index in C.
        
    Returns
    -------
    teleport_idx : np.array, shape [S]
                   The indicies of the nodes in the teleport set.
    """
    #### YOUR CODE ####
    # Extract the column indices of 'C' from 'categories' by relevant categories in topics
    c_col_idx = [categories[c] for c in topics]
    
    # Compute the vector that contains number of categories each restaurant has
    num_cat = np.sum(C[:,c_col_idx], axis=1)

    # get the teleport indices by extracting indices of restaurants having nonzero number of categories
    teleport_idx = np.flatnonzero(num_cat)
    
    return teleport_idx


# In[21]:


teleport_idx = teleport_set(C, ['Mexican'], categories)
print(teleport_idx)


#  ## 3. Implement topic-sensitive PageRank

# In[8]:


def page_rank(A, beta, teleport_idx=None, eps=1e-12):
    """
    Implements topic-sensitive PageRank using power iteration and sparse matrix operations.
    
    Parameters
    ----------
    A           : sp.spmatrix, shape [num_restaurants, num_restaurants]
                  The adjacency matrix representing the graph of restaurants.
    beta        : float, 
                  0 < beta < 1, (1-beta) is the probabilty of teleporting to the nodes in the teleport set
    teleport_idx: np.array, shape [S]
                  The indices of the nodes in the teleport set. If it equals to None
                  it means runs standard PageRank, i.e. all nodes are in the teleport set.
    
    Returns
    -------
    r          : np.array, shape [num_restaurants]
                 The page rank vector containing the page rank scores for each restaurant.
    """
    
    #### YOUR CODE ####
    # number of restaurants
    n = A.shape[0]

    # initialize r
    r = np.random.uniform(size=n)
    r = r / np.sum(r)

    # computes penalty for standard PageRank OR
    # computes penalty only for restaurants in teleport set and the rest of restaurants gets 0 teleport prob.
    penalty = np.zeros(n)
    if teleport_idx is None:
        penalty = (1 - beta) * np.full(n, 1 / n)
    else:
        penalty[teleport_idx] = (1 - beta) * 1 / len(teleport_idx)

    # makes A a column stochastic matrix (normalizing)
    A_array = A.toarray()
    A_normalized = A_array / A_array.sum(axis=0, keepdims=True)
    A = sp.csr_matrix(A_normalized)

    # initialize r_prev to store r in the previous iteration
    r_prev = np.zeros(n)

    # computes importance rank iteratively until reaching stationary distribution
    while np.linalg.norm(
            r - r_prev) >= eps:  # the Euclidean distance between r_updated and r must be less than eps for the loop to stop
        r_prev = r
        r = beta * A.dot(r) + penalty

    return r


# ### 3.1 Calculate the standard PageRank scores and print the names of the top 5 restaurants overall

# In[9]:


idx_to_category = {v:k for k, v in categories.items()}


# In[10]:


r = page_rank(A=A, beta=0.6, teleport_idx=None)

for i, x in enumerate(r.argsort()[-5:]):
    print(i+1, names[x], '\n  Categories: ', [idx_to_category[cat] for cat in C[x].nonzero()[1]])


# ### 3.2 Calculate the topic-sensitive PageRank scores and print the names of top 5 Mexican restaurants

# In[11]:


teleport_idx = teleport_set(C, ['Mexican'], categories)
r = page_rank(A=A, beta=0.6, teleport_idx=teleport_idx)

for i, x in enumerate(r.argsort()[-5:]):
    print(i+1, names[x], '\n  Categories: ', [idx_to_category[cat] for cat in C[x].nonzero()[1]])


# ### 3.3 Calculate the topic-sensitive PageRank scores and print the names of top 5 Italian or French restaurants
# 

# In[12]:


teleport_idx = teleport_set(C, ['Italian', 'French'], categories)
r = page_rank(A=A, beta=0.6, teleport_idx=teleport_idx)

for i, x in enumerate(r.argsort()[-5:]):
    print(i+1, names[x], '\n  Categories: ', [idx_to_category[cat] for cat in C[x].nonzero()[1]])


# In[ ]:




