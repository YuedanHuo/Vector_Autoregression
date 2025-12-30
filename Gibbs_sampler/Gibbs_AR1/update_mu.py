import numpy as np


def update_mu(trajectory_k, Phi_k, rho_k, mu_0 = 0 , sigma0_2 = np.inf):
    """
    Update the kth element of mu (asymptotic mean of AR(1) log-volatility).
    
    trajectory_k : array of shape (T,), log-lambda path including h_0
    Phi_k        : variance of AR(1) innovations
    rho_k        : AR(1) coefficient
    mu_0         : prior mean
    sigma0_2     : prior variance
    """
    T = trajectory_k.shape[0]

    # y_t = h_t - rho * h_{t-1}
    y = trajectory_k[1:] - rho_k * trajectory_k[:-1]
    sum_y = np.sum(y)

    # posterior variance
    post_var = 1.0 / (((1 - rho_k)**2) * T / Phi_k + 1.0 / sigma0_2)

    # posterior mean
    post_mean = post_var * ((1 - rho_k) * sum_y / Phi_k + mu_0 / sigma0_2)

    return np.random.normal(post_mean, np.sqrt(post_var))


import numpy as np

def update_mu_vectorized(trajectory, rho, Phi, mu_0 = 0 , sigma0_2 = np.inf):
    """
    Vectorized update of mu for all K series.
    
    trajectory : array of shape (K, T)
    rho        : array of shape (K,)
    Phi        : array of shape (K,)
    mu_0       : scalar or array of shape (K,)
    sigma0_2   : scalar or array of shape (K,)
    
    Returns:
        mu : array of shape (K,)
    """
    K, T = trajectory.shape

    # Compute y_t = h_t - rho * h_{t-1} for all series
    y = trajectory[:, 1:] - np.multiply(trajectory[:, :-1].T, rho).T  # shape (K, T)
    sum_y = np.sum(y, axis=1)  # shape (K,)

    # Posterior variance and mean
    post_var = 1.0 / (((1 - rho)**2) * T / Phi + 1.0 / sigma0_2)
    post_mean = post_var * ((1 - rho) * sum_y / Phi + mu_0 / sigma0_2)

    # Draw mu for all series
    mu = np.random.normal(post_mean, np.sqrt(post_var))

    return mu
