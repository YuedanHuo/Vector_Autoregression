from scipy.stats import invgamma

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
