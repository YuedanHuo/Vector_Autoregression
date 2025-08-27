import numpy as np

def update_Pi_corrected(Ay, z, Pi, log_lambdas, lags, A, prior_mean, inv_prior_var):
    """
    Update each row of the coefficient matrix Pi using the revised algorithm.

    Parameters:
    y            : ndarray (T, N) - Observed time series data.
    z            : ndarray (T, k) - Lagged data, including intercept.
    Pi           : ndarray (N, k) - Current coefficient estimates.
    log_lambdas  : ndarray (T, N) - Log of the volatility terms for each variable.
    A            : ndarray (N, N) - Cholesky factor of Σ_t.
    prior_mean   : ndarray (N, k) - Prior mean for each row of Pi.
    prior_var    : list of ndarrays (N,) - Prior covariance for each row of Pi.

    Returns:
    Pi : ndarray (N, k) - Updated coefficient matrix.
    """
    T, N = Ay.shape
    k = z.shape[1]

    for j in range(N):
        # Initialize prior precision and the data term (X'X)
        prior_prec = inv_prior_var[j]  # Inverse of the prior variance
        data_prec = np.zeros_like(prior_prec)  # Will accumulate the sum from all equations
        data_contrib = np.zeros(k)  # Initialize data contribution for posterior mean

        # Calculate residuals for the j-th equation: Ay - Pi @ A^T
        A_zero_diag = A.copy()
        np.fill_diagonal(A_zero_diag, 0) # as we don't want to subtract the diagonal
        residuals_j = Ay - np.dot(z, Pi.T) @ A_zero_diag.T  # Residuals estimation from previous draw

        # Residuals_j should be scaled by λ (volatility)
        residuals_j_scaled = residuals_j[:, j:] / np.exp(0.5 * log_lambdas[j:,:].T)  # Columns j to N

        # Scale z_j by volatility
        z_j = z[:, j:]
        scale_factor = 1 / np.exp(0.5 * log_lambdas[j:, :].T)

        # Accumulate data precision (sum of outer products scaled by A[j:, j])
        for i in range(j, N):
            data_prec += residuals_j_scaled[:, i-j].T@residuals_j_scaled[:, i-j] * A[j, i]**2
            data_contrib += np.sum((z_j[:,i-j]*scale_factor[:,i-j] * A[j, i]).T@residuals_j_scaled[:, :], axis=0)

        # Posterior precision and covariance
        post_prec = prior_prec + data_prec
        post_prec += np.eye(post_prec.shape[0]) * 1e-6  # Regularization
        post_cov = np.linalg.pinv(post_prec)

        # Posterior mean
        post_mean = post_cov @ (prior_prec @ prior_mean[j] + data_contrib)

        # Sample from the posterior distribution for the j-th row of Pi
        Pi[j, :] = np.random.multivariate_normal(post_mean, post_cov)

    return Pi
