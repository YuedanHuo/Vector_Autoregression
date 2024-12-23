from joblib import Parallel, delayed

p =13 # set up lag

#set up prior
bayesian_var_prior = BayesianVARPrior(y, p=p, sigma_0=2.31, threshold=0.5)
priors = bayesian_var_prior.get_priors()
#for A
Sigma_A = priors["Sigma_A"]
mu_A = priors["mu_A"]
#for Pi
Pi_prior_mean = priors["Pi_prior_mean"]
Pi_prior_var_rows = priors["Pi_prior_var"]
Pi_prior_var_rows_inv = priors["Pi_prior_var_inv"]
# for log_lambdas
sigma_0 = priors["sigma_0"]
phi = priors["phi"]
# finally the data
z_test = priors["Z"]
y_test = priors["y"]
T,N = y_test.shape


n_iter = 2500  # number of iterations

def run_gibbs_sampler(sampler, n_iter):
    sampler.run(n_iter)
    return sampler  # Return the sampler for accessing results later.

# Create the two samplers
gibbs_sampler = GibbsSampler(
    y=y_test,
    z=z_test,
    mu_A=mu_A,
    Sigma_A=Sigma_A,
    Pi_prior_mean=Pi_prior_mean,
    Pi_prior_var=Pi_prior_var_rows,
    Pi_prior_var_inv=Pi_prior_var_rows_inv,
    phi=phi,
    sigma0=sigma_0,
    T=T,
    N=N,
    p=p,
    Num=30
)

gibbs_sampler_oldf = GibbsSampler_oldf(
    y=y_test,
    z=z_test,
    mu_A=mu_A,
    Sigma_A=Sigma_A,
    Pi_prior_mean=Pi_prior_mean,
    Pi_prior_var=Pi_prior_var_rows,
    Pi_prior_var_inv=Pi_prior_var_rows_inv,
    phi=phi,
    sigma0=sigma_0,
    T=T,
    N=N,
    p=p,
    Num=15  # number of steps in MH
)

results = Parallel(n_jobs=2)(  # n_jobs specifies the number of parallel tasks
    delayed(run_gibbs_sampler)(sampler, n_iter)
    for sampler in [gibbs_sampler, gibbs_sampler_oldf]
)

# Access results
for completed_sampler in results:
    print(f"Completed sampler: {completed_sampler}")


A_csmc = np.array(results[0].samples["A"])
A_oldf = np.array(results[1].samples["A"])
Pi_csmc = np.array(results[0].samples["Pi"])
Pi_oldf = np.array(results[1].samples["Pi"])
log_lambdas_csmc = np.array(results[0].samples["log_lambdas"])
log_lambdas_oldf = np.array(results[1].samples["log_lambdas"])
Phi_csmc = np.array(results[0].samples["Phi"])
Phi_oldf = np.array(results[1].samples["Phi"])
