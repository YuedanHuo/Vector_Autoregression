import time
import numpy as np
import matplotlib as plt

n_iter = 10
Ns = [5,10,15,20,25,30,35,40]
time_csmc = np.empty((n_iter, len(Ns)))
time_oldf = np.empty((n_iter, len(Ns)))
p = 3

for n in Ns:
  # trim y for the number of variables
  y_test = y[:,:n]
  print(y_test.shape)
  #set up prior
  bayesian_var_prior = BayesianVARPrior(y_test, p=p, sigma_0=2.31, threshold=0.5)
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

  gibbs_sampler = GibbsSampler(
    y=y_test,
    z=z_test,
    mu_A=mu_A,
    Sigma_A=Sigma_A,
    Pi_prior_mean=Pi_prior_mean,
    Pi_prior_var=Pi_prior_var_rows,
    Pi_prior_var_inv=Pi_prior_var_rows_inv,
    phi= np.ones(N),
    sigma0=sigma_0,
    T=T,
    N=N,
    p=p,
    Num=30
  )
  for i in range(n_iter):
    start_time = time.time()
    gibbs_sampler.run(10)
    end_time = time.time()
    time_csmc[i,Ns.index(n)] = end_time - start_time

  gibbs_sampler_oldf = GibbsSampler_oldf(
    y=y_test,
    z=z_test,
    mu_A=mu_A,
    Sigma_A=Sigma_A,
    Pi_prior_mean=Pi_prior_mean,
    Pi_prior_var=Pi_prior_var_rows,
    Pi_prior_var_inv=Pi_prior_var_rows_inv,
    phi= np.ones(N),
    sigma0=sigma_0,
    T=T,
    N=N,
    p=p,
    Num=40 # number of step in MH
  )
  for i in range(n_iter):
    start_time = time.time()
    gibbs_sampler_oldf.run(10)
    end_time = time.time()
    time_oldf[i,Ns.index(n)] = end_time - start_time

plt.plot(Ns, time_csmc.mean(axis = 0), label='csmc')
plt.plot(Ns, time_oldf.mean(axis = 0), label='carriero')
plt.xlabel('N')
plt.ylabel('Time (seconds)')
plt.title('Time to run GibbsSampler (CSMC) vs GibbsSampler (MHMC)')
plt.legend()
plt.show()