import numpy as np
from numba import njit

@njit
def update_APi_numba(Ay, z, Pi, log_lambdas, lags, A, prior_mean, inv_prior_var, first_iter=False):
    """
    Numba-optimized version of update_APi with vectorized matrix operations where possible.

    Parameters:
        Ay            : (T, N)
        z             : (T, k)
        Pi            : (N, k)
        log_lambdas   : (N, T)
        A             : (N, N)
        prior_mean    : (N, k)
        inv_prior_var : (N, k, k)  - precomputed inverse covariance for each row
    """
    T, N = Ay.shape
    k = z.shape[1]
    Pi_new = np.empty((N, k))
    
    for j in range(N):
        # Scale factor for heteroskedasticity
        scale_factor = 1.0 / np.exp(log_lambdas[j, :])
        weighted_z = z * scale_factor[:, None]  # shape (T, k)

        # Posterior precision
        data_prec = np.ascontiguousarray(weighted_z.T) @ np.ascontiguousarray(z)
        #data_prec = weighted_z.T @ z
        post_prec = data_prec + inv_prior_var[j]
        post_prec += np.eye(k) * 1e-6  # regularization

        # Solve for posterior covariance
        L = np.linalg.cholesky(post_prec)
        post_cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))


        # Conditional prior mean
        prev_contrib = prev_contrib = A[j, :j].copy() @ Pi_new[:j, :].copy() # sum of previous rows
        con_prior_mean = prior_mean[j] + prev_contrib

        # Posterior mean
        data_contrib = np.ascontiguousarray(weighted_z.T) @ np.ascontiguousarray(Ay[:, j])
        post_mean = post_cov @ (inv_prior_var[j] @ con_prior_mean + data_contrib)

        # Sample from posterior
        L = np.linalg.cholesky(post_cov)
        eps = np.random.randn(k)
        APi = post_mean + L @ eps

        Pi_new[j, :] = APi - prev_contrib

    return Pi_new
