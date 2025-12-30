import numpy as np


def check_pi_stability(pi_matrix, N, p):
    """
    Checks the stability of a VAR(p) model from its coefficient matrix Pi.

    Args:
        pi_matrix (np.ndarray): The (K, K*p + 1) coefficient matrix.
        N (int): Number of variables.
        p (int): VAR order.

    Returns:
        bool: True if stable, False otherwise.
    """
    if p == 0:
        return True
    
    # The companion matrix is formed from Pi excluding the intercept column
    pi_coeffs = pi_matrix[:, 1:]
    
    companion_matrix = np.zeros((N * p, N * p))
    companion_matrix[:N, :] = pi_coeffs
    if p > 1:
        companion_matrix[N:, :-N] = np.eye(N * (p - 1))
    
    eigenvalues = np.linalg.eigvals(companion_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    
    print(f"Maximum eigenvalue modulus: {max_eigenvalue}")
    
    return max_eigenvalue < 1, max_eigenvalue


def update_APi (Ay, z, Pi, log_lambdas, lags, A, prior_mean, inv_prior_var, first_iter = False):
    """
    Update each row of the coefficient matrix Pi using the revised algorithm.

    Parameters:
    y            : ndarray (T, N) - Observed time series data.
    z            : ndarray (T, k) - Lagged data, including intercept.
    Pi           : ndarray (N, k) - Current coefficient estimates.
    log_lambdas  : ndarray (N, T) - Log of the volatility terms for each variable.
    A            : ndarray (N, N) - Cholesky factor of Σ_t.
    prior_mean   : ndarray (N, k) - Prior mean for each row of Pi.
    prior_var    : list of ndarrays (N,) - Prior covariance for each row of Pi.
    first_iter   : bool - If this is the first iteration after initializing from Prior.

    Returns:
    Pi : ndarray (N, k) - Updated coefficient matrix.
    """
    T, N = Ay.shape
    k = z.shape[1]
    Pi = np.zeros((N,k))
    
    for j in range(N):
      # Initialize prior precision and the data term (X'X)
      prior_prec = inv_prior_var[j]  # Inverse of the prior variance, also the conditional variance
        
      # calculate the posterior covariance
      scale_factor = 1 / np.exp(log_lambdas[j, :])
      weighted_z = z * scale_factor[:, None]  # broadcasting over columns
      data_prec = weighted_z.T @ z
      post_prec = data_prec + prior_prec
      post_prec += np.eye(post_prec.shape[0]) * 1e-6  # Regularization
      #post_cov = np.linalg.pinv(post_prec)
      L = np.linalg.cholesky(post_prec)
      post_cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))


      # calculate the conditional prior
      prev_contrib = A[j,:j] @ Pi[:j, :]
      con_prior_mean = prior_mean[j] + prev_contrib
      data_contrib = weighted_z.T @ Ay[:, j]
      post_mean = prior_prec @ con_prior_mean + data_contrib  ## potential issue with the prior_prec here
      post_mean = post_cov @ post_mean

      # Sample from the posterior distribution for the j-th row of APi
      #APi= np.random.multivariate_normal(post_mean, post_cov)
      L = np.linalg.cholesky(post_cov) # use chol decompostion for faster computation
      APi = post_mean + L @ np.random.randn(k)
      Pi[j,:] = APi - prev_contrib


    return Pi