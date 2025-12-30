
import numpy as np
from numba import njit

@njit
def compute_posterior_A_with_log_lambdas_numba(y, z, log_lambdas, mu_A, Sigma_A, Pi):
    T, N = y.shape
    A_draw = np.eye(N)
    y_tilde = y - z @ Pi.T

    for j in range(1, N):
        y_tilde_j = y_tilde[:, :j]
        inv_Sigma_A = np.linalg.solve(Sigma_A[:j, :j], np.eye(j))
        Sigma_A_new = inv_Sigma_A.copy()

        for t in range(T):
            Sigma_A_new += np.outer(y_tilde_j[t, :], y_tilde_j[t, :]) * np.exp(-log_lambdas[j, t])
        
        Sigma_A_new += 1e-8 * np.eye(j)
        Sigma_A_new = np.linalg.solve(Sigma_A_new, np.eye(j))
        Sigma_A_new = 0.5 * (Sigma_A_new + Sigma_A_new.T)

        mu_A_new = inv_Sigma_A @ mu_A[:j]
        for t in range(T):
            mu_A_new += y_tilde_j[t, :] * np.exp(-log_lambdas[j, t])
        mu_A_new = Sigma_A_new @ mu_A_new

        # Multivariate normal sample via Cholesky
        L = np.linalg.cholesky(Sigma_A_new)
        A_draw[j, :j] = mu_A_new + L @ np.random.randn(j)

    return A_draw
