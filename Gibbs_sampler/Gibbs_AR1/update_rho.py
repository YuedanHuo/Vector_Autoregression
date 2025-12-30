import numpy as np
from scipy.stats import norm

def log_prior_rho(rho, phi1, phi2):
    # Beta(phi1, phi2) on (rho+1)/2
    if rho <= -1 or rho >= 1:
        return -np.inf
    u = (rho + 1) / 2
    return (phi1 - 1) * np.log(u) + (phi2 - 1) * np.log(1 - u)

def update_rho(trajectory_k, Phi_k, mu_k, phi1, phi2, rho_curr):
    """
    MH update for rho_k with Beta(phi1, phi2) prior on (rho+1)/2
    trajectory_k : (T,) log lambda path for series k 
    Phi_k        : variance of AR(1) error
    mu_k         : mean of AR(1) process
    phi1, phi2   : Beta prior params
    rho_curr     : current rho value
    """
    h = trajectory_k
    x = h[:-1] - mu_k
    y = h[1:] - mu_k

    # OLS estimate and variance (proposal mean and var)
    sum1 = np.sum(x**2)
    sum2 = np.sum(x*y)
    rho_hat = sum2 / sum1
    var_rho = Phi_k / sum1

    # propose rho from Normal(rho_hat, var_rho)
    rho_prop = np.random.normal(rho_hat, np.sqrt(var_rho))


    # log-priors
    lp_curr = log_prior_rho(rho_curr, phi1, phi2)
    lp_prop = log_prior_rho(rho_prop, phi1, phi2)

    # MH ratio
    # only account for the prior as proposal and likelihood cancel out
    log_accept = lp_prop - lp_curr

    if np.log(np.random.rand()) < log_accept:
        return rho_prop
    else:
        return rho_curr
