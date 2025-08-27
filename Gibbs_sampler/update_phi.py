from scipy.stats import invgamma
import numpy as np

def compute_posterior_phi(log_lambdas,df_phi, sigma_phi=1):
    N, T = log_lambdas.shape  # Get dimensions
    # Vectorized computation of sum_of_difference over all time periods
    sum_of_difference = np.sum((log_lambdas[:, 1:] - log_lambdas[:, :-1])**2, axis=1)

    # Set the new degrees of freedom and scale parameters
    new_df = np.ones(N) * (df_phi + T)  # Degrees of freedom for each dimension
    new_sigma = sigma_phi + sum_of_difference  # Updated scale based on computed differences

    # Sample new phi values using inverse-gamma for all dimensions
    new_phi = invgamma.rvs(a=new_df / 2, scale=new_sigma / 2)

    return new_phi

def compute_posterior_phi_by_component(log_lambdas_list, df_phi_vec, K):
    """
    log_lambdas_list: list of length K, each element is a 1D NumPy array of length T_j + 1
    df_phi_vec: 1D NumPy array of length K (degrees of freedom per variable)
    K: number of variables
    """
    phi_samples = []

    for j in range(K):
        logs = log_lambdas_list[j]
        T_j = df_phi_vec[j]

        diffs = logs[1:] - logs[:-1]
        S = np.sum(diffs ** 2)

        alpha = (K + 2 + T_j) / 2
        beta = (1.0 + S) / 2

        phi_j = 1.0 / (np.random.gamma(alpha) * beta)
        phi_samples.append(phi_j)

    return np.array(phi_samples)