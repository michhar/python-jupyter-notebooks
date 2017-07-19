import numpy as np
from random import seed
import math

class AdalineSGD:
    """Adaptive Linear Neuron classifier with sigmoid activation.

    Parameters
    ----------

    eta : float
        Learning rate ([0,1])
    niter : int
        Iterations on training dataset (epochs)

    Attributes
    ----------

    w_ : 1d-array
        Weights post-fitting
    errors_ : list
        Count of misclassifications in each epoch
    shuffle : bool (default: True)
        If True, shuffle data each epoch to avoid
        cycles.
    random_state : int (default: None)
        Random state used for shuffling and
        initializing the weights.
        
    """

    def __init__(self, eta=0.01, niter=10, shuffle=True,
                 random_state=None):
        self.eta = eta
        self.niter = niter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
           seed(random_state)

    def fit(self, X, y):
        """Fit the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (n-dimensional - represented as
            a matrix), where n_samples is the number of
            data points/samples and n_features is the number
            of features.
        y : array-like, shape (n_samples)
            Target labels/values (1d-array)

        Returns
        -------
        self : object

        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.niter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            mean_cost = np.mean(cost)
            self.cost_.append(mean_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit the training data without reinitializing the
        weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        """Calculate net input into network"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute sigmoidal activation
        
        Returns
        -------
        A 1d array of length n_samples

        """
        x = self.net_input(X)
        func = lambda v: 1 / (1 + math.exp(-v))
        return np.array(list(map(func, x)))

    def predict(self, X):
        """Predict class label after applying activation function"""
        return np.where(self.activation(X) >= 0.5, 1, -1)

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zero"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply the adaline learning rule to update the
        weights."""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = error**2 * 0.5
        return cost


