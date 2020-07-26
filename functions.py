import numpy as np
def sigmoid(z):

    """
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
       sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))