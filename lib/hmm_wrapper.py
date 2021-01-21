import numpy as np
from numba import njit
from numba import prange

import lib.fun as fun

class binomial_HMM():
    """
    Multidimensional Hidden Markov Model with D independent binomial emission,
    and N hidden states.

    This class wraps the functions defined in lib/fun.py that use numba.
    """
    def __init__(self, N, D):
        self.N = N
        self.D = D

        # Initialize the parameters of the HMM to some random values
        state_priors = np.random.rand(N)
        state_priors /= state_priors.sum(axis = 0)
        self.log_state_priors = np.log(state_priors)

        transition_matrix = np.diag(np.ones(N)) + np.random.rand(N,N)
        transition_matrix /= transition_matrix.sum(axis = 1)[:, None]
        self.log_transition_matrix = np.log(transition_matrix)

        emission_matrix = np.random.rand(N,D)
        self.log_emission_matrix = np.log(emission_matrix)

        self.best_likelihood = -np.inf

    def update(self, x):
        """
        -----------------------------------------------------------------
        Arguments:  - self
                    - x, 1D numpy array of shape T
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        This function wraps up all the functions defined in lib/fun.py
        and performs the M-step, updating the parameters of the model
        given the observation sequence.
        -----------------------------------------------------------------
        """

        log_emission_probabilities = fun.emission_model(self.log_emission_matrix, x)
        log_alpha, log_beta = fun.forward_backward(log_emission_probabilities, self.log_transition_matrix,
                                                   self.log_state_priors)

        log_gamma = fun.evaluate_log_gamma(log_alpha, log_beta)
        log_xi = fun.evaluate_log_xi(log_gamma, log_beta, self.log_transition_matrix,
                                     log_emission_probabilities)

        self.log_state_priors, self.log_transition_matrix, self.log_emission_matrix = fun.M_step(log_gamma, log_xi,
                                                                                                 self.log_transition_matrix, x)

        likelihood = fun.logsumexp(log_alpha[-1], axis = 0)
        print('\t Current likelihood: ' + "{:.2f}".format(likelihood))
        if likelihood > self.best_likelihood:
            self.best_likelihood = likelihood
            print('Best likelihood!')

            self.states_probability = np.exp(log_gamma)
