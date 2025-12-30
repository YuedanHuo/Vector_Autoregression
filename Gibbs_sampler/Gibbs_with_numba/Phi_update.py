from numba import njit
import numpy as np

@njit(nogil=True)
def sample_inverse_gamma(alpha, beta):
    """
    Sample from Inverse-Gamma(alpha, beta) using the standard Gamma transform.
    X ~ InvGamma(alpha, beta) <=> X = 1 / Y, Y ~ Gamma(alpha, 1/beta)
    """
    return 1.0 / np.random.gamma(alpha, 1.0 / beta)


@njit(nogil=True)
def compute_posterior_phi_by_component_numba(fix_log_lambda_matrix, valid_lengths, K):
    """
    Computes posterior Phi samples using a fixed 2D matrix instead of a list.
    
    Args:
        fix_log_lambda_matrix (float[:, :]): 2D Array of shape (K, Max_T). 
                                             Contains the log-lambda paths.
        valid_lengths (int[:]): 1D Array of shape (K,). 
                                Represents the 'df' (degrees of freedom) or effective 
                                length of the path for each component k.
        K (int): Number of components.
        
    Returns:
        phi_samples (float[:]): Array of sampled Phi values.
    """
    phi_samples = np.empty(K)
    
    for j in range(K):
        # 1. Retrieve the effective length for this component
        # This replaces logs.shape[0]
        T_limit = valid_lengths[j]
        
        # 2. Compute Sum of Squared Differences
        # We access the matrix directly. Row j, indices up to T_limit.
        S = 0.0
        
        # We start from 1 because we need (t) - (t-1).
        # We iterate up to T_limit. 
        # Example: If path has length 5 (indices 0,1,2,3,4), T_limit is 5.
        # Loop runs 1, 2, 3, 4.
        #logs = fix_log_lambda_matrix[j, :T_limit]
        #diffs = logs[1:] - logs[:-1]
        #S = np.sum(diffs ** 2)

        for t in range(0, T_limit-1):
            diff = fix_log_lambda_matrix[j, t+1] - fix_log_lambda_matrix[j, t]
            S += diff * diff

        # 3. Posterior Hyperparameters
        # Note: 'valid_lengths[j]' is used here as T_j (the degree of freedom)
        alpha = (K + 2 + T_limit) / 2.0
        beta = (1.0 + S) / 2.0

        # 4. Inverse Gamma Sampling
        # Logic: X ~ Gamma(alpha, 1). Result = 1 / (X * beta)
        # 1 / (X * beta) is equivalent to InvGamma(alpha, beta)
        gamma_sample = np.random.gamma(alpha)
        phi_samples[j] = 1.0 / (gamma_sample * beta)

    return phi_samples

@njit(nogil=True)
def compute_posterior_phi_numba(log_lambdas_list, df_phi_vec, K):
    """
    Fully Numba-compatible posterior phi update.
    log_lambdas_list: list of length K, each element is 1D array of length T_j + 1
    df_phi_vec: array of degrees of freedom per variable
    K: number of variables
    """
    phi_samples = np.empty(K)
    
    for j in range(K):
        logs = log_lambdas_list[j]
        T_j = df_phi_vec[j]

        diffs = logs[1:] - logs[:-1]
        S = np.sum(diffs ** 2)

        alpha = (K + 2 + T_j) / 2.0
        beta = (1.0 + S) / 2.0

        phi_samples[j] = sample_inverse_gamma(alpha, beta)

    return phi_samples

