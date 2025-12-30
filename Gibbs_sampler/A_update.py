# this update is specifically for the prior on A such that each row independent
# And have the same variance and mean but different dimension
# As we define it in the prior of A
import numpy as np

def compute_posterior_A_with_log_lambdas(y, z, log_lambdas, mu_A, Sigma_A, Pi):
    """
    Compute the posterior mean and covariance for A in a VAR model with heteroskedastic errors,
    using log-transformed Lambda values (log_lambdas), y, and z.

    Parameters:
    y (numpy.ndarray): The data matrix of shape (T, N), where T is the number of time periods, and N is the number of variables.
    z (numpy.ndarray): The exogenous variable matrix of shape (T, M), where M is the number of exogenous variables.
    log_lambdas (numpy.ndarray): The log-transformed volatility matrix of shape (N, T), where each column represents log(Lambda_t).
    mu_A (numpy.ndarray): Prior mean for a row of A, of shape (N ,).
    Sigma_A (numpy.ndarray): Prior covariance matrix for a row of A, of shape (N, N).
    Pi (numpy.ndarray): The coefficient matrix Pi of shape (N, M), used to transform z.
    T (int): Number of time periods (rows of y and z).

    Returns:
    mu_A_prime (numpy.ndarray): Posterior mean for A, of shape (N * N,).
    Sigma_A_prime (numpy.ndarray): Posterior covariance matrix for A, of shape (N * N, N * N).
    """

    T,N = y.shape  # Number of variables
    A_draw = np.eye(N)


    # Transform y_t using the formula y_t_tilde = y_t - Pi @ z_t
    y_tilde = y - z @ Pi.T  # Shape: (T, N)

    for j in range(1,N): # do the update independently for each row of A


      # for the jth row, we only need the following y_tilde value
      # as A^{-1} is lower triangular, with diagonal value 1
      y_tilde_j = y_tilde[:, :j]

      # update the covariance matrix
      inv_Sigma_A = np.linalg.solve(Sigma_A[:j,:j], np.eye(j))
      Sigma_A_new = inv_Sigma_A
      for t in range(T):
        Sigma_A_new += np.outer(y_tilde_j[t, :], y_tilde_j[t, :]) * np.exp(-log_lambdas[j, t])
      eps = 1e-8
      Sigma_A_new += eps * np.eye(j)
      Sigma_A_new = np.linalg.solve(Sigma_A_new, np.eye(j))
      Sigma_A_new = 0.5 * (Sigma_A_new + Sigma_A_new.T) # ensure symmetricity


      # update the mean
      mu_A_new = inv_Sigma_A @ mu_A[:j]
      for t in range(T):
        mu_A_new += y_tilde_j[t, :] * np.exp(-log_lambdas[j, t])
      mu_A_new = Sigma_A_new @ mu_A_new
      A_draw[j,:j] = np.random.multivariate_normal(mu_A_new, Sigma_A_new)

    return A_draw


import numpy as np

def compute_posterior_A_with_log_lambdas_vectorized(y, z, log_lambdas, mu_A, Sigma_A, Pi):
    """
    Compute posterior draw for A in a VAR with heteroskedastic errors.
    A is assumed lower-triangular with ones on the diagonal.
    """

    T, N = y.shape
    A_draw = np.eye(N)
    y_tilde = y - z @ Pi.T  # (T, N)

    for j in range(1, N):
        # Regressor part (only previous variables since A is lower-triangular)
        y_tilde_j = y_tilde[:, :j]  # (T, j)

        # Prior precision and prior mean (restricted to dimension j)
        Sigma_A_j = Sigma_A[:j, :j]
        prec_prior = np.linalg.solve(Sigma_A[:j,:j], np.eye(j))
        mean_prior = mu_A[:j]

        # Compute observation precision weights
        w = np.exp(-log_lambdas[j, :])  # (T,)

        # Weighted cross-product
        # Σ_t w_t y_t y_t' = (Y' diag(w) Y)
        YW = y_tilde_j * np.sqrt(w)[:, None]
        prec_lik = YW.T @ YW  # (j, j)

        # Posterior precision and covariance
        prec_post = prec_prior + prec_lik
        # Small regularization for numerical stability
        prec_post += 1e-8 * np.eye(j)
        Sigma_A_new = np.linalg.solve(prec_post, np.eye(j))
        Sigma_A_new = 0.5 * (Sigma_A_new + Sigma_A_new.T)

        # Posterior mean
        mu_post = prec_prior @ mean_prior + y_tilde_j.T @ (w * np.ones_like(w))
        mu_post = Sigma_A_new @ mu_post

        #Sample A[j, :j] ~ N(mu_post, Sigma_A_new)
        #A_draw[j, :j] = mu_post + np.linalg.cholesky(Sigma_A_new +  1e-8 * np.eye(j)) @ np.random.randn(j)
        A_draw[j,:j] = np.random.multivariate_normal(mu_post, Sigma_A_new)

    return A_draw
