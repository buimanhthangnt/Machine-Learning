import numpy as np
from models.gradient_check import eval_numerical_gradient

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D = W.shape[0]
    N = X.shape[0]
    diff = np.zeros_like(y)
    
    for i in range(N):
        product = 0
        for j in range(D):
            product += X[i][j] * W[j]
        diff[i] = product - y[i]
        loss += diff[i] ** 2    
    for i in range(D):
        loss += reg * W[i] * W[i]
    loss = loss / (2 * N)
    
    #gradiens
    for i in range(D):
        for j in range(N):
            dW[i] += X[j][i] * diff[j]
        dW[i] += reg * W[i]
        dW[i] = dW[i] / N 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    diff = X.dot(W) - y
    loss = (np.sum(diff ** 2) + np.sum(reg * W ** 2)) / (2 * len(y))
    dW = (np.transpose(X).dot(diff) + reg * W) / len(y)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW