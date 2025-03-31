import numpy as np


# SIGMOID function
def sigmoid(z):
    """ The sigmoid function maps any input to a value between 0 and 1. It is commonly used for binary classification tasks,
      especially in the output layer..."""
    return 1/(1+np.exp(-z))

# ReLU function - Rectified linear Unit
def relu(z):
    """ The ReLU function outputs the input directly if it is positive; otherwise, it will output zero. 
    It is one of the most commonly used activation functions for hidden layers..."""
    return np.maximum(0,z)

# Leaky ReLU: solving the problem of "dying neurons" in ReLU
def Leaky_ReLU(z,alpha=0.01):
    """Leaky ReLU allow small slope for negative values, solving the problem of "dying neurons" in ReLU. 
    The small slope is determined by a hyperparameter 𝛼 """
    return np.where(z>0,z,alpha*z)

# tanh function 
def tanh(z):
    """ The tanh function maps any input to value between -1 and 1. 
    It is similar to sigmoid but is zero-centered.
    """
    return np.tanh(z)

# SoftMax Function
def softmax(z):
    """ The Softmax function is used to normalize the outputs of a multi-class classification model. 
    It converts raw scores (logits) into probabilities that sum to 1. This is typically used in the output layer 
    for multi-class classification tasks...
    """
    exp_z=np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z,axis=0)

# Swish Function
def swish(z):
    """
    Swish is a self-gated activation function. It is a newer activation function and has been shown to outperform ReLU in some situations,
    especially in deeper networks.
    """
    return z*sigmoid(z)


