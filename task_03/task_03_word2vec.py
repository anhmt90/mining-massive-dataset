#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import tensorflow as tf
from scipy import spatial
# get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Introduction
# 
# The goal of this project is to obtain the vector representations for words from text.
# 
# The main idea is that words appearing in similar contexts have similar meanings. Because of that, word vectors of similar words should be close together. Models that use word vectors can utilize these properties, e.g., in sentiment analysis a model will learn that "good" and "great" are positive words, but will also generalize to other words that it has not seen (e.g. "amazing") because they should be close together in the vector space.
# 
# Vectors can keep other language properties as well, like analogies. The question "a is to b as c is to ...?", where the answer is d, can be answered by looking into word vector space and calculating $\mathbf{u}_b - \mathbf{u}_a + \mathbf{u}_c$, and finding the word vector that is the closest to the result.
# 
# We are given a text that contains $N$ unique words $\{ x_1, ..., x_N \}$. We will focus on the Skip-Gram model in which the goal is to predict the context window $S = \{ x_{i-l}, ..., x_{i-1}, x_{i+1}, ..., x_{i+l} \}$ from current word $x_i$, where $l$ is the window size. 
# 
# We get a word embedding $\mathbf{u}_i$ by multiplying the matrix $\mathbf{U}$ with a one-hot representation $\mathbf{x}_i$ of a word $x_i$. Then, to get output probabilities for context window, we multiply this embedding with another matrix $\mathbf{V}$ and apply softmax. The objective is to minimize the loss: $-\mathop{\mathbb{E}}[P(S|x_i;\mathbf{U}, \mathbf{V})]$.
# 
# You are given a dataset with positive and negative reviews. Your task is to:
# + Construct input-output pairs corresponding to the current word and a word in the context window
# + Implement forward and backward propagation with parameter updates for Skip-Gram model
# + Train the model
# + Test it on word analogies and sentiment analysis task

# # 2. Load data
# 
# We'll be working with a subset of reviews for restaurants in Las Vegas. The reviews that we'll be working with are either 1-star or 5-star. You can download the used data set (`task03_data.npy`) from:
# 
# * ([download link](https://syncandshare.lrz.de/dl/fi7cjApuE3Bd3xyfsyx3k9jr/task03_data.npy)) the preprocessed set of 1-star and 5-star reviews 

# In[2]:


data = np.load("task03_data.npy", allow_pickle=True)
reviews_1star = [[x.lower() for x in s] for s in data.item()["reviews_1star"]]
reviews_5star = [[x.lower() for x in s] for s in data.item()["reviews_5star"]]


# We generate the vocabulary by taking the top 500 words by their frequency from both positive and negative sentences. We could also use the whole vocabulary, but that would be slower.

# In[3]:


vocabulary = [x for s in reviews_1star + reviews_5star for x in s]
vocabulary, counts = zip(*Counter(vocabulary).most_common(500))


# In[4]:


VOCABULARY_SIZE = len(vocabulary)
EMBEDDING_DIM = 100


# In[5]:


print('Number of positive reviews:', len(reviews_1star))
print('Number of negative reviews:', len(reviews_5star))
print('Number of unique words:', VOCABULARY_SIZE)


# You have to create two dictionaries: `word_to_ind` and `ind_to_word` so we can go from text to numerical representation and vice versa. The input into the model will be the index of the word denoting the position in the vocabulary.

# In[6]:


"""
Implement
---------
word_to_ind: dict
    The keys are words (str) and the value is the corresponding position in the vocabulary
ind_to_word: dict
    The keys are indices (int) and the value is the corresponding word from the vocabulary
ind_to_freq: dict
    The keys are indices (int) and the value is the corresponding count in the vocabulary
"""
word_to_ind = defaultdict(int)
ind_to_word = defaultdict(str)
ind_to_freq = defaultdict(int)
for xi, word in enumerate(vocabulary):
    word_to_ind[word] = xi
    ind_to_word[xi] = word
    ind_to_freq[xi] = counts[xi]
    
### YOUR CODE HERE ###


# In[7]:


print('Word \"%s\" is at position %d appearing %d times' % 
      (ind_to_word[word_to_ind['the']], word_to_ind['the'], ind_to_freq[word_to_ind['the']]))


# In[8]:


word_to_ind['']


# # 3. Create word pairs
# 
# We need all the word pairs $\{ x_i, x_j \}$, where $x_i$ is the current word and $x_j$ is from its context window. These will correspond to input-output pairs. We want them to be represented numericaly so you should use `word_to_ind` dictionary.

# In[9]:


def get_window(sentence, window_size):
    sentence = [x for x in sentence if x in vocabulary]
    pairs = []

    """
    Iterate over all the sentences
    Take all the words from (i - window_size) to (i + window_size) and save them to pairs
    
    Parameters
    ----------
    sentence: list
        A list of sentences, each sentence containing a list of words of str type
    window_size: int
        A positive scalar
        
    Returns
    -------
    pairs: list
        A list of tuple (word index, word index from its context) of int type
    """

    ### YOUR CODE HERE ###
    pairs = []
    
    for xi,centerWord in enumerate(sentence):
        theRange = xi 
        if theRange > window_size:
            theRange = window_size
        if theRange !=0:  
            for backward in range(theRange):
                pairs.append((word_to_ind[centerWord], word_to_ind[sentence[xi - backward -1 ]]))
                
        theRange = len(sentence)-1-xi
        if theRange > window_size:
            theRange = window_size

        if theRange !=0:  
            for forward in range(theRange):
                pairs.append((word_to_ind[centerWord], word_to_ind[sentence[xi + 1 + forward]]))


            
    #pair = list(pairs)        
    return pairs


# In[10]:


data = []
for x in reviews_1star + reviews_5star:
    data += get_window(x, window_size=3)
data = np.array(data)

print('First 5 pairs:', data[:5].tolist())
print('Total number of pairs:', data.shape[0])


# We calculate a weighting score to counter the imbalance between the rare and frequent words. Rare words will be sampled more frequently. See https://arxiv.org/pdf/1310.4546.pdf

# In[11]:


probabilities = [1 - np.sqrt(1e-3 / ind_to_freq[x]) for x in data[:,0]]
probabilities /= np.sum(probabilities)


# # 4. Model definition
# 
# In this part you should implement forward and backward propagation together with update of the parameters.

# In[12]:


class Embedding():
    def __init__(self, N, D, seed=None):
        """
        Parameters
        ----------
        N: int
            Number of unique words in the vocabulary
        D: int
            Dimension of the word vector embedding
        seed: int
            Sets the random seed, if omitted weights will be random
        """

        self.N = N
        self.D = D
        
        self.init_weights(seed)
    
    def init_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        """
        We initialize weight matrices U and V of dimension (D, N) and (N, D) respectively
        """
        self.U = np.random.normal(0, np.sqrt(2 / self.D / self.N), (self.D, self.N))
        self.V = np.random.normal(0, np.sqrt(2 / self.D / self.N), (self.N, self.D))

    def one_hot(self, x, N):
        """
        Given a vector returns a matrix with rows corresponding to one-hot encoding
        
        Parameters
        ----------
        x: array
            M-dimensional vector containing integers from [0, N]
        N: int
            Number of posible classes
        
        Returns
        -------
        one_hot: array
            (N, M) matrix where each column is N-dimensional one-hot encoding of elements from x 
        """

        ### YOUR CODE HERE ###
        M = x.shape[0]
        one_hot = np.zeros((N, M))
        for i in range(M):
            if(ind_to_word.get(i) != None):
                one_hot[x[i], i] = 1
        
#         for xi,intWord in enumerate(x):
#             one_hot[xi,intWord] = 1
            

        assert one_hot.shape == (N, x.shape[0])
        return one_hot

    def loss(self, y, prob):
        """
        Parameters
        ----------
        y: array
            (N, M) matrix of M samples where columns are one-hot vectors for true values
        prob: array
            (N, M) column of M samples where columns are probabily vectors after softmax

        Returns
        -------
        loss: int
            Cross-entropy loss calculated as: 1 / M * sum_i(sum_j(y_ij * log(prob_ij)))
        """

        ### YOUR CODE HERE ###
        M = y.shape[1]
        dot = np.multiply(y, np.log(prob))
        loss = -1/M * np.sum(np.sum(dot, axis=0))
#         loss = -1/M * np.
        return loss
    
    def softmax(self, x, axis):
        """
        Parameters
        ----------
        x: array
            A non-empty matrix of any dimension
        axis: int
            Dimension on which softmax is performed
            
        Returns
        -------
        y: array
            Matrix of same dimension as x with softmax applied to 'axis' dimension
        """
        
        ### YOUR CODE HERE ###
        exp = np.exp(x)
        
        # each column adds up to 1
        if axis == 0:
            y = exp/np.sum(exp, axis=0)
        
        # each row adds up to one
        if axis == 1:
            y = np.transpose(np.transpose(exp)/np.sum(exp, axis=1))
                   
        return y
    
    def step(self, x, y, learning_rate=1e-3):
        """
        Performs forward and backward propagation and updates weights
        
        Parameters
        ----------
        x: array
            M-dimensional mini-batched vector containing input word indices of int type
        y: array
            Output words, same dimension and type as 'x'
        learning_rate: float
            A positive scalar determining the update rate
            
        Returns
        -------
        loss: float
            Cross-entropy loss
        d_U: array
            Partial derivative of loss w.r.t. U
        d_V: array
            Partial derivative of loss w.r.t. V
        """
        
        # Input transformation
        """
        Input is represented with M-dimensional vectors
        We convert them to (N, M) matrices such that columns are one-hot 
        representations of the input
        """
        x = self.one_hot(x, self.N)
        y = self.one_hot(y, self.N)

        
        # Forward propagation
        """
        Returns
        -------
        embedding: array
            (D, M) matrix where columns are word embedding from U matrix
        logits: array
            (N, M) matrix where columns are output logits
        prob: array
            (N, M) matrix where columns are output probabilities
        """
        
        ### YOUR CODE HERE ###
        #Omran:
        #U and V of dimension (D, N) and (N, D) respectively

        embedding  = np.dot(self.U, x)
        logits  = np.dot(self.V, embedding)
        prob = self.softmax(logits,0)# take care of the axis, I am not quite sure how you will implement it
            
        assert embedding.shape == (self.D, x.shape[1])
        assert logits.shape == (self.N, x.shape[1])
        assert prob.shape == (self.N, x.shape[1])
    
    
        # Loss calculation
        """
        Returns
        -------
        loss: int
            Cross-entropy loss using true values and probabilities
        """
        
        ### YOUR CODE HERE ###
        loss = self.loss(y, prob)
        
        # Backward propagation
        """
        Returns
        -------
        d_U: array
            (N, D) matrix of partial derivatives of loss w.r.t. U
        d_V: array
            (D, N) matrix of partial derivatives of loss w.r.t. V
        """
        
        ### YOUR CODE HERE ###
        #I am not quite sure of this!!
                   
#         difference = np.sum(np.subtract(prob, y), axis=1)
        difference = prob - y
        d_V = difference @ embedding.T
#         print(self.N, self.D)
#         print(difference.shape)
#         print(d_V.shape)
        d_U = (self.V.T @ difference) @ x.T
#         d_U = self.V.T @ np.outer(difference, x)
                   
        assert d_V.shape == (self.N, self.D)
        assert d_U.shape == (self.D, self.N)
 
        
        # Update the parameters
        """
        Updates the weights with gradient descent such that W_new = W - alpha * dL/dW, 
        where alpha is the learning rate and dL/dW is the partial derivative of loss w.r.t. 
        the weights W
        """
        
        ### YOUR CODE HERE ###
        self.V = self.V - learning_rate * d_V
        self.U = self.U - learning_rate * d_U

        return loss, d_U, d_V


# ## 4.1 Gradient check
# 
# The following code checks whether the updates for weights are implemented correctly. It should run without an error.

# In[13]:


def get_loss(model, old, variable, epsilon, x, y, i, j):
    delta = np.zeros_like(old)
    delta[i, j] = epsilon

    model.init_weights(seed=132) # reset weights
    setattr(model, variable, old + delta) # change one weight by a small amount
    loss, _, _ = model.step(x, y) # get loss

    return loss

def gradient_check_for_weight(model, variable, i, j, k, l):
    x, y = np.array([i]), np.array([j]) # set input and output
    
    old = getattr(model, variable)
    
    model.init_weights(seed=132) # reset weights
    _, d_U, d_V = model.step(x, y) # get gradients with backprop
    grad = { 'U': d_U, 'V': d_V }
    
    eps = 1e-4
    loss_positive = get_loss(model, old, variable, eps, x, y, k, l) # loss for positive change on one weight
    loss_negative = get_loss(model, old, variable, -eps, x, y, k, l) # loss for negative change on one weight
    
    true_gradient = (loss_positive - loss_negative) / 2 / eps # calculate true derivative wrt one weight

    assert abs(true_gradient - grad[variable][k, l]) < 1e-5 # require that the difference is small

def gradient_check():
    N, D = VOCABULARY_SIZE, EMBEDDING_DIM
    model = Embedding(N, D)

    # check for V
    for _ in range(20):
        i, j, k = [np.random.randint(0, d) for d in [N, N, D]] # get random indices for input and weights
        gradient_check_for_weight(model, 'V', i, j, i, k)

    # check for U
    for _ in range(20):
        i, j, k = [np.random.randint(0, d) for d in [N, N, D]]
        gradient_check_for_weight(model, 'U', i, j, k, i)

    print('Gradients checked - all good!')

gradient_check()


# # 5. Training

# We train our model using stochastic gradient descent. At every step we sample a mini-batch from data and update the weights.
# 
# The following function samples words from data and creates mini-batches. It subsamples frequent words based on previously calculated probabilities.

# In[14]:


def get_batch(data, size, prob):
    i = np.random.choice(data.shape[0], size, p=prob)
    return data[i, 0], data[i, 1]


# Training the model can take some time so plan accordingly.

# In[15]:


# np.random.seed(123)
# model = Embedding(N=VOCABULARY_SIZE, D=EMBEDDING_DIM)
#
# losses = []
#
# MAX_ITERATIONS = 150000
# PRINT_EVERY = 10000
#
# for i in range(MAX_ITERATIONS):
#     x, y = get_batch(data, 128, probabilities)
#     loss, _, _ = model.step(x, y, 1e-3)
#     losses.append(loss)
#
#     if (i + 1) % PRINT_EVERY == 0:
#         print('Iteration:', i + 1, 'Loss:', np.mean(losses[-PRINT_EVERY:]))


# In[16]:


import pickle

# model_out = open("model.pickle","wb")
# losses_out = open("losses.pickle","wb")

# pickle.dump(losses, model_out)
# pickle.dump(losses, losses_out)
# model_out.close()
# losses_out.close()


# In[ ]:


model = pickle.load(open("model.pickle","rb"))
losses = pickle.load(open("losses.pickle","rb"))


# The embedding matrix is given by $\mathbf{U}^T$, where the $i$th row is the vector for $i$th word in the vocabulary.

# In[17]:


emb_matrix = model.U.T


# # 6. Analogies
# 
# As mentioned before, vectors can keep some language properties like analogies. Given a relation a:b and a query c, we can find d such that c:d follows the same relation. We hope to find d by using vector operations. In this case, finding the real word vector $\mathbf{u}_d$ closest to $\mathbf{u}_b - \mathbf{u}_a + \mathbf{u}_c$ gives us d.

# In[18]:


triplets = [['go', 'going', 'come'], ['look', 'looking', 'come'], ['you', 'their', 'we'], 
            ['what', 'that', 'when'], ['go', 'went', 'is'], ['go', 'went', 'find']]

for triplet in triplets:
    a, b, c = triplet

    """
    Returns
    -------
    candidates: list
        A list of 5 closest words, measured with cosine similarity, to the vector u_b - u_a + u_c
    """

    ### YOUR CODE HERE ###
    x = np.array([word_to_ind[a], word_to_ind[b], word_to_ind[c]])
    inputs = model.one_hot(x, model.N)
    u = emb_matrix.T @ inputs
    # print(u.shape)

    u_d = u[:, 1] - u[:, 0] + u[:, 2]

    cosine_distances = []
    for word in vocabulary:
        one_hot = model.one_hot(np.array([word_to_ind[word]]), model.N)
        u_i = emb_matrix.T @ one_hot
        cosine_distances.append((word_to_ind[word], spatial.distance.cosine(u_d, u_i)))

    # for v_i in model.V:
    #     cosine_distances.append((word_to_ind[word], spatial.distance.cosine(u_d, v_i)))

    cosine_distances.sort(key=lambda tup: tup[1])
    candidates_indices = [i[0] for i in cosine_distances[:5]]

    candidates = []
    for i in candidates_indices:
        candidates.append(ind_to_word[i])

    # result = 1 - spatial.distance.cosine(ux, ui[])

    print('%s is to %s as %s is to [%s]' % (a, b, c, '|'.join(candidates)))


# # RNN
#
# Our end goal is to use the pretrained word vectors on some downstream task, e.g. sentiment analysis. We first generate a dataset where we just concatenate 1 and 5-star reviews into `all_sentences`. We also create a list `Y` with labels 1 for positive reviews and 0 for negative

# In[ ]:


all_sentences = reviews_1star + reviews_5star
Y = np.array([0] * len(reviews_1star) + [1] * len(reviews_5star))

SENTENCES_SIZE = len(all_sentences)
MAX_SENTENCE_LENGTH = max([len(x) for x in all_sentences])


# Your task is to create an array $\mathbf{X}$ where (i,j,k) element denotes $k$th value of an embedding for $j$th word in $i$th sentence in the dataset. In addition, we need a list that keeps track of how many words are in each sentence.

# In[ ]:


"""
Returns
-------
X: array
    Array of dimensions (SENTENCES_SIZE, MAX_SENTENCE_LENGTH, EMBEDDING_DIM) where
    the first dimension denotes the index of the sentence in the dataset and second is
    the word index in the sentence. Sentences that are shorter than MAX_SENTENCE_LENGTH
    are padded with zero vectors. Words that are not in the vocabulary are also
    represented with zero vectors of EMBEDDING_DIM size.
S: array
    Array of SENTENCES_SIZE dimension containing the sentence lenghts
"""

### YOUR CODE HERE ###
X = np.zeros((SENTENCES_SIZE, MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
for i, sentence in enumerate(all_sentences):
    indices = []
    for word in sentence:
        indices.append(word_to_ind[word])

    one_hot_mat = model.one_hot(np.array(indices), model.N)
    result = (model.U @ one_hot_mat).T
    result = np.pad(result, ((0,MAX_SENTENCE_LENGTH - len(sentence)),(0,0)), 'constant')
    X[i] = result

S = [len(x) for x in all_sentences]
#
#
# # We want to train on a subset of data, and test on remaining data. Your task is to split X, Y and S into training and test set (60%-40%).
#
# # In[ ]:
#
#
# """
# Returns
# -------
# X_train, y_train, s_train: arrays
#     Randomly selected 60% of all data
# X_test, y_test, s_test: arrays
#     Rest of the data
# """
#
# ### YOUR CODE HERE ###
# BATCH_SIZE = 1
#
#
# # LSTM implementation in tensorflow. Inputs are padded sequences of word vectors, sentence lengths, and true labels (0 or 1). The model takes word vectors and passes them through the LSTM. Final state is used as an input of one fully connected layer with output dimension 1. We also get probability that the class is positive and argmax label. Network uses Adam optimizer.
#
# # In[ ]:
#
#
# #Omran: I think this section is done, if you face any problem let me know
# class LSTM:
#     def __init__(self, cell_dim=64):
#         """
#         Attributes
#         ----------
#         x: float
#             Input sentence of shape (BATCH SIZE, MAX SENTENCE LENGTH, EMBEDDING DIM)
#         y: float
#             Output label of shape (BATCH SIZE)
#         s: float
#             Length of sentences of shape (BATCH SIZE)
#         last_state: float
#             The last state of sequences with shape (BATCH SIZE, CELL DIM)
#         logits: float
#             The
#         prob: float
#             Probabilities after sigmoid
#         y_hat: int
#             Predicted class value (0 or 1)
#         loss: float
#             Cross entropy loss
#         optimize:
#             Operation that updates the weights based on the loss
#         accuracy: float
#             Accuracy of prediction y_hat given y
#         """
#
#
#         """
#         Define input placeholders x, y and s as class attributes
#         """
#         ### YOUR CODE HERE ###
#         #Omran: the first dim is none so batch size can  take any value.
#         self.x = tf.placeholder(tf.float64, [None, MAX_SENTENCE_LENGTH,EMBEDDING_DIM], name='x')
#         self.y = tf.placeholder(tf.float64, [None], name='y')
#         self.s = tf.placeholder(tf.float64, [None], name='s')
#
#         """
#         Use dynamic_rnn to define an LSTM layer
#         Define last_state as class attribute to be the last output h of LSTM
#         (Note that we have zero padding)
#         """
#         ### YOUR CODE HERE ###
#
#         self.lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_dim)
#         init_state = cells.zero_state(BATCH_SIZE, tf.float32)
#
#         #I am not quite sure if we need outputs, which all the states outputs.
#         outputs, self.last_state = tf.nn.dynamic_rnn(self.lstm_cell,
#             rnn_inputs, initial_state=init_state, sequence_length=self.s)
#         """
#         Define logits, prob and y_hat as class attributes.
#         We get logits by applying a single dense layer on the last state.
#         """
#         ### YOUR CODE HERE ###
#         self.logits = tf.layers.dense(
#             inputs=l,
#             units=2,
#             activation=None,
#             name="output_hidden_layer")
#
#
#         self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))
#         self.optimize = tf.train.AdamOptimizer().minimize(self.loss)
#
#         self.y_hat = tf.argmax(out, axis=1)
#         correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#         self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#
# # In this part we finally train our RNN model and evaluate on the test set.
#
# # In[ ]:
#
#
# tf.reset_default_graph()
# tf.set_random_seed(123)
# np.random.seed(123)
#
# model = LSTM()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for iter in range(300):
#         i = np.random.randint(0, X_train.shape[0], 64)
#         feed = { model.x: X_train[i], model.y: y_train[i], model.s: s_train[i] }
#         _ = sess.run(model.optimize, feed)
#
#         if (iter + 1) % 100 == 0:
#             train_loss, train_accuracy = sess.run([model.loss, model.accuracy], feed)
#             print('Iter:', iter + 1, 'Train loss:', train_loss, 'Train accuracy:', train_accuracy)
#
#     test_loss, test_pred = sess.run([model.loss, model.y_hat], { model.x: X_test, model.y: y_test, model.s: s_test })
#     print('Test loss:', test_loss, 'Test accuracy:', np.mean(test_pred == y_test))


# In[ ]:




