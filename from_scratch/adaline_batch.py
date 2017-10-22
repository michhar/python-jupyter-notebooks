import numpy as np

class AdalineGD:
    """Adaptive Linear Neuron classifier.

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

    """

    def __init__(self, eta=0.01, niter=50):
        self.eta = eta
        self.niter = niter

    def fit(self, X, y):
        """Fit the training data.

        Parameters
        ----------
        X : nd array-like, shape = [n_samples, n_features]
            Training data (1-n dimensional - represented as
            a matrix), where n_samples is the number of
            data points/samples and n_features is the number
            of features.
        y : 1d array-like, shape = [n_samples]
            Target labels/values (1d-array)

        Returns
        -------
        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        # Iterate over data (all data processed each time)
        for i in range(self.niter):
            output = self.net_input(X)

            # True label minus output value
            errors = (y - output)

            # Update weights based on sum of all errors
            self.w_[1:] += self.eta * X.T.dot(errors)

            # Update bias based on sum of all errors
            self.w_[0] += self.eta * errors.sum()

            # Cost function
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """Calculate net input into network"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Predict class label after applying linear activation function"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

