#!/usr/bin/env python
# coding: utf-8

# # Project task 01: Near duplicate detection with LSH

# In[1]:


import gzip
import tarfile

import numpy as np
import pandas as pd
import time

from sklearn import preprocessing
from collections import defaultdict

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# To goal of this task is to find near duplicate songs in the Million Song dataset. You can imagine a scenario were the same song appears on multiple different releases with only small feature variation (e.g. duration or loudness).

# ## 1. Load data and extract the data

# We'll be working with the Million Songs Dataset, a freely-available collection of audio features and metadata for a million contemporary popular music tracks.
# 
# Specifically, we will work with a smaller subset of 10 000 songs ([download link](http://static.echonest.com/millionsongsubset_full.tar.gz)).

# In[2]:


tar = tarfile.open('millionsongsubset_full.tar.gz', 'r')
members = tar.getmembers()


# In[3]:


tar.extract(members[5])
summary = pd.HDFStore(members[5].name)
songs = summary['/analysis/songs']


# Show a snippet of how the data looks like:

# In[4]:


songs.head()


# We should have $31$ columns and $10~000$ rows.

# In[5]:


print(len(songs))


# Since not all features are important we are going to consider a subset of features (columns) that are relevant for duplicate detection.
# 
# We will also convert the pandas dataframe into a numpy array so it is easier to work with.

# In[6]:


subset = songs[['duration', 'end_of_fade_in', 'key', 'loudness',
                'mode', 'start_of_fade_out', 'tempo', 'time_signature',]]

data_matrix = subset.values


# Additionally we will standardize the data to have zero mean and unit variance as a preprocessing step.

# In[7]:


scaled_data = preprocessing.scale(data_matrix)


# ## 2. Implementaion
# 
# Your task is to implement near duplicate detection using LSH with cosine similarity.
# More specifically you have to:
# * Generate duplicate **candidates** based on LSH with $b$ bands and $r$ rows per band
# * Refine the candidates by computing the exact cosine distance
# * Report all pairs/duplicates with cosine distance < $d$

# Implement a function that computes the cosine distance between two rows (instances) in the data.

# In[8]:


def cosine_distance(X, i, j):
    """Compute cosine distance between two rows of a data matrix.
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix.
    i : int
        Index of the first row.
    j : int
        Index of the second row.
        
    Returns
    -------
    d : float
        Cosine distance between the two rows of the data matrix.
        
    """
    d = None
    
    ### YOUR CODE HERE ###
    
    return d


# Cosine distance between the 5-th and the 28-th instance

# In[9]:


print('{:.4f}'.format(cosine_distance(scaled_data, 5, 28)))


# In[10]:


def LSH(X, b=8, r=32, d=0.3):
    """Find candidate duplicate pairs using LSH and refine using exact cosine distance.
    
    Parameters
    ----------
    X : np.array shape [N, D]
        Data matrix.
    b : int
        Number of bands.
    r : int
        Number of rows per band.
    d : float
        Distance treshold for reporting duplicates.
    
    Returns
    -------
    duplicates : {(ID1, ID2, d_{12}), ..., (IDX, IDY, d_{xy})}
        A set of tuples indicating the detected duplicates.
        Each tuple should have 3 elements:
            * ID of the first song
            * ID of the second song
            * The cosine distance between them
    
    n_candidates : int
        Number of detected candidate pairs.
        
    """
    np.random.seed(158)
    n_candidates = 0
    duplicates = set()

    ### YOUR CODE HERE ###
    
    return duplicates, n_candidates


# In[11]:


duplicates, n_candidates = LSH(scaled_data, b=3, r=64, d=0.0003)


# In[12]:


print('We detected {} candidates.'.format(n_candidates))


# Show the duplicates we have found:

# In[13]:


duplicates


# Show the metadata for the songs that were detected as duplicates:

# In[14]:


for i, j, d in duplicates:
    print('Song ID 1: {}'.format(i),
          'Song ID 2: {}'.format(j),
          'Distance: {:.6f}'.format(d),
          summary['/metadata/songs'].loc[i][['title', 'artist_name']].str.cat(sep=' - '),
          summary['/metadata/songs'].loc[j][['title', 'artist_name']].str.cat(sep=' - '), sep='\n')
    print()


# ## 3. Compare runtime

# Your task is to implement code for runtime comparison between LSH and the naive nested for loop implementation.

# In[15]:


# naively compute the duplicates using a double for loop
def naive_duplicates(X, d = 0.2):
    """
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix.
    d : float
        Distance treshold for reporting duplicates.
    
    Returns
    -------
    duplicates : {(ID1, ID2, d_{12}), ..., (IDX, IDY, d_{xy})}
        A set of tuples indicating the detected duplicates.
        Each tuple should have 3 elements:
            * ID of the first song
            * ID of the second song
            * The cosine distance between them
    """
    N = X.shape[0]
    duplicates = set()
    for i in range(N):
        for j in range(N):
            d_ij = cosine_distance(X, i, j)
            if d_ij < d and i != j:
                duplicates.add((i, j, d_ij))
    return duplicates


# In[16]:


def runtime_comparison():
    """
    Compare the runtime between LSH and the naive approach.
    
    Returns
    -------
    trace : [(n1, lsh_dur, naive_dur), (n2, lsh_dur, naive_dur), ... ]
            A list of tuples with execution times for different number of songs.
            Each tuple should have 3 elements:
                * number of songs considered
                * duration of the LSH approach
                * duration of the naive approach
    """
    trace = []
    for n in np.arange(25, 251, 25):
        print('Running comparison for {} songs.'.format(n))
        
        ### YOUR CODE HERE ###
        
    return trace


# In[17]:


trace = runtime_comparison()


# Plot the differecene in runtime. On the x-axis plot the number of songs processed and on the y-axis plot the runtime in seconds for both approaches. You should obtain a plot similar to the one shown below.

# In[18]:


### YOUR PLOTTING CODE HERE ###

