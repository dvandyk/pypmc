"""Collect MCMC proposal densities"""

import numpy as np
from scipy.special import gammaln

class ProposalDensity(object):
    """A proposal density for a local-random-walk Markov chain sampler."""

    def evaluate(self, x, y):
        """Evaluate log of the density to propose ``x`` given ``y``, namely log(q(x|y)).

        :param x:

            The proposed point.

        :param y:

            The current point of the Markov chain.

        """
        raise NotImplementedError()

    def propose(self, y, rng = 'numpy.random.mtrand'):
        """Propose a new point given ``y`` using the random number
        generator ``rng``.

        :param y:

            The current position of the chain.

        :param rng:

            The state of a random number generator like numpy.random.mtrand

        .. important::
            ``rng`` must return a numpy array of N samples from: \n
            - **rng.normal(0,1,N)**: standard gaussian distribution
            - **rng.chisquare(degree_of_freedom, N)**: any chi-squared distribution

        """
        raise NotImplementedError()

class AdaptiveProposal(ProposalDensity):
    """Abstract proposal density with adaptation"""

    def adapt(self, points):
        """Adapt the proposal function based on a sequence of chain
        samples.

        :param points:

            An iterable of points from the chain in the order
            generated by the chain.

        """
        raise NotImplementedError()

class Multivariate(AdaptiveProposal):
    """A multivariate proposal density with covariance adaptation and
    rescaling

    """

    def __init__(self, sigma):
        self.dim   = sigma.shape[0]
        self.sigma = sigma.copy()

        self._sigma_decompose()


    def _sigma_decompose(self):
        """Private function to calculate the Cholesky decomposition, the
        inverse and the normalisation of the covariance matrix sigma and
        store it in the object instance

        """
        self.cholesky_sigma = np.linalg.cholesky(self.sigma)
        self.inv_sigma      = np.linalg.inv(self.sigma)
        self._compute_norm()

    def _compute_norm(self):
        """Private function to calculate the normalisation of the
        covariance matrix sigma and store it in the object instance

        """
        raise NotImplementedError()

    def _get_gauss_sample(self, rng):
        """transform sample from standard gauss to Gauss(mean=0, sigma = sigma)"""
        return np.dot(self.cholesky_sigma,rng.normal(0,1,self.dim))

class MultivariateGaussian(Multivariate):
    """A multivariate Gaussian density with covariance adaptation and rescaling

    :param sigma:

         A numpy array representing the covariance-matrix.

    """

    def _compute_norm(self):
        self.log_normalization = -.5 * self.dim * np.log(2 * np.pi) + .5 * np.log(np.linalg.det(self.inv_sigma))

    def evaluate(self, x , y):
        return self.log_normalization - .5 * np.dot(np.dot(x-y, self.inv_sigma), x-y)

    def propose(self, y, rng = np.random.mtrand):
        return y + self._get_gauss_sample(rng)

class MultivariateStudentT(Multivariate):
    """A multivariate Student-t density with covariance adaptation and rescaling

    :param sigma:

         A numpy array representing the covariance-matrix.


    :param dof:

         A float or int representing the degree of freedom.

    """

    def __init__(self, sigma, dof):
        self.dof = dof
        super(MultivariateStudentT, self).__init__(sigma)

    def _compute_norm(self):
        self.log_normalization = gammaln(.5 * (self.dof + self.dim)) - gammaln(.5 * self.dof) \
                                 -0.5 * self.dim * np.log(self.dof * np.pi) + .5 * np.log(np.linalg.det(self.inv_sigma))

    def evaluate(self, x , y):
        return self.log_normalization  - .5 * (self.dof + self.dim) \
            * np.log(1. + (np.dot(np.dot(x-y, self.inv_sigma), x-y)) / self.dof)

    def propose(self, y, rng = np.random.mtrand):
        # when Z is normally distributed with expected value 0 and std deviation sigma
        # and  V is chi-squared distributed with dof degrees of freedom
        # and  Z and V are independent
        # then Z*sqrt(dof/V) is t-distributed with dof degrees of freedom and std deviation sigma

        return y + self._get_gauss_sample(rng) * np.sqrt(self.dof / rng.chisquare(self.dof))
