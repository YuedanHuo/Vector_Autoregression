import numpy as np

def sample_lambdas(u, lambda_current, phi, index, num_iter=1):
    """
    Sample stochastic volatilities using the Jacquier et al. (1994) Metropolis-Hastings method.

    Parameters:
    u               : np.array of shape (T,), Ay_t - A@Pi@z_t, for some j in {1,...,N}
    lambda_current       : np.array of shape (T,), current volatility states
    phi  :          array of shape(N,)
    num_iter        : int, number of MH iterations (default=1)

    Returns:
    lambda_new           : np.array of shape (T,), updated volatility states
    """
    T = len(u)
    lambda_new = lambda_current.copy()
    sigma_c2 = 0.5 * phi[index]  # Variance for the log-volatility proposal

    for t in range(T):
        # Compute conditional mean of the log-volatility
        if t == 0:  # Edge case for the first time point
            mu_t = np.log(lambda_new[t + 1])
        elif t == T - 1:  # Edge case for the last time point
            mu_t = np.log(lambda_new[t - 1])
        else:
            mu_t = 0.5 * (np.log(lambda_new[t - 1]) + np.log(lambda_new[t + 1]))

        for _ in range(num_iter):  # Repeat for MH iterations
            # Propose a new h_t
            log_lambda_proposal = np.random.normal(mu_t, np.sqrt(sigma_c2))
            lambda_proposal = np.exp(log_lambda_proposal)
            lambda_proposal = np.clip(lambda_proposal, 1e-12, 1e12)

            # Calculate the acceptance probability
            current_likelihood = -0.5 * (u[t]**2 / lambda_new[t]) - 0.5 * np.log(lambda_new[t])
            proposed_likelihood = -0.5 * (u[t]**2 / lambda_proposal) - 0.5 * log_lambda_proposal

            acceptance_ratio = proposed_likelihood - current_likelihood

            # Accept or reject the proposal
            if np.log(np.random.uniform(0, 1)) < acceptance_ratio:
                lambda_new[t] = lambda_proposal

    return lambda_new
