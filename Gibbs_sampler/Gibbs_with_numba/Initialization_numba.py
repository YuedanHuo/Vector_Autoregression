import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
from joblib import Parallel, delayed

from A_update_numba import compute_posterior_A_with_log_lambdas_numba
from Phi_update import compute_posterior_phi_numba
from CSMC import CSMC
from APi_update import update_APi_numba
from Gibbs import GibbsSampler_API

sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from prior_setup import BayesianVARPrior
from SMC import SMC


import numpy as np
import scipy
import time

from joblib import Parallel, delayed

def _initialize_one_theta_chain(i, y, z, mu_A, Sigma_A, Pi_prior_mean, Pi_prior_var_rows,
                                Pi_prior_var_rows_inv, phi, sigma_0, p, N_x, K, N_rejuvenate, N_burn_in=500):
    """
    Initialize one θ-particle via Gibbs + CSMC.
    """
    init_t = y.shape[0]
    y_init = y
    z_init = z

    gibbs_sampler = GibbsSampler_API(
        y=y_init,
        z=z_init,
        mu_A=mu_A,
        Sigma_A=Sigma_A,
        Pi_prior_mean=Pi_prior_mean,
        Pi_prior_var=Pi_prior_var_rows,
        Pi_prior_var_inv=Pi_prior_var_rows_inv,
        phi=phi,
        sigma0=sigma_0,
        T=init_t,
        N=K,
        p=p,
        Num=N_x
    )

    gibbs_sampler.run(N_rejuvenate + 1 + N_burn_in)

    A_list, Pi_list, Phi_list, Api_list, = [], [], [], [], 
    trajectories = np.zeros((N_rejuvenate + 1, N_x, K, init_t + 1))
    ancestors = np.zeros((N_rejuvenate + 1, N_x, K, init_t), dtype=int)

    for j in range(N_rejuvenate + 1):
        A = gibbs_sampler.samples['A'][N_burn_in + j]
        Pi = gibbs_sampler.samples['Pi'][N_burn_in + j]
        Phi = gibbs_sampler.samples['phi'][N_burn_in + j]
        Api = A @ Pi
        ancestor = gibbs_sampler.samples['ancestors'][N_burn_in + j]
        trajectory = gibbs_sampler.samples['trejactories'][N_burn_in + j]
        
        trajectories[j, :, :, 1:init_t + 1] = trajectory
        ancestors[j, :, :, :init_t] = ancestor

        A_list.append(A)
        Pi_list.append(Pi)
        Phi_list.append(Phi)
        Api_list.append(Api)


    return {
        'A': A_list,
        'Pi': Pi_list,
        'Phi': Phi_list,
        'Api': Api_list,
        'trajectories': trajectories,
        'ancestors': ancestors
    }

def initialize_from_posterior(y, p, init_t, N_rejuvenate, N_x, M_theta):
    bayesian_var_prior = BayesianVARPrior(y, p=p, sigma_0=2.31, threshold=0.5)
    priors = bayesian_var_prior.get_priors()

    Sigma_A = priors["Sigma_A"]
    mu_A = priors["mu_A"]
    Pi_prior_mean = priors["Pi_prior_mean"]
    Pi_prior_var_rows = priors["Pi_prior_var"]
    Pi_prior_var_rows_inv = priors["Pi_prior_var_inv"]
    sigma_0 = priors["sigma_0"]
    phi = priors["phi"]
    z = priors["Z"]
    y = priors["y"]
    K = y.shape[1]

    init_y = y[:init_t, :]
    init_z = z[:init_t, :]

    results = Parallel(n_jobs=-1)(
        delayed(_initialize_one_theta_chain)(
            i, init_y, init_z, mu_A, Sigma_A,
            Pi_prior_mean, Pi_prior_var_rows, Pi_prior_var_rows_inv,
            phi, sigma_0, p, N_x, K, N_rejuvenate
        ) for i in range(M_theta)
    )

    return results
