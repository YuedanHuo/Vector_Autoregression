import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
sys.path.append('/files/Gibbs_sampler')
from prior_setup import BayesianVARPrior
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi, compute_posterior_phi_by_component
from SMC import SMC
sys.path.append('/files/Gibbs_sampler/CSMC_gibbs')
from CSMC_bs import CSMC
sys.path.append('/files/Gibbs_sampler/APi_Gibbs')
from Update_APi import update_APi
from Gibbs_Api import GibbsSampler_API
sys.path.append('/files/experiment')

import numpy as np
import scipy
import time

# Note it is ok to pass full obs y into the rejuvenation of A and Pi
# as the unobserved log_lambdas is initialize as positive  inf

class SMC_Square:
    def __init__(self, p, logfile, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.5, threshold_A = 0.75):
        self.p = p
        self.N_theta = N_theta
        self.N_x = N_x
        self.threshold = threshold
        self.threshold_A = threshold_A # set two threshold
        self.N_rejuvenate = N_rejuvenate
        self.logfile = logfile
        self.need_rejuvenate_A = False

    def log(self, message):
        with open(self.logfile, 'a') as f:
            f.write(message + '\n')
            
    def init_prior(self, y):

        bayesian_var_prior = BayesianVARPrior(y, p=self.p, sigma_0=2.31, threshold=0.5)
        priors = bayesian_var_prior.get_priors()

        self.Sigma_A = priors["Sigma_A"]
        self.mu_A = priors["mu_A"]
        self.Pi_prior_mean = priors["Pi_prior_mean"]
        self.Pi_prior_var_rows = priors["Pi_prior_var"]
        self.Pi_prior_var_rows_inv = priors["Pi_prior_var_inv"]
        self.sigma_0 = priors["sigma_0"]
        self.z = priors["Z"]
        self.y = priors["y"]
        self.Phi = priors['phi']
        self.T, self.K = self.y.shape

        self.outer_weights = np.zeros((self.N_theta,))
        self.inner_weights = np.zeros((self.N_theta, self.N_x, self.K)) # K independent chains 
        #self.log_lambdas = np.full((self.N_theta, self.N_x, self.K, self.T + 1), np.inf)
        self.trajectories = np.zeros((self.N_theta, self.N_x, self.K, self.T + 1))
        self.ancestors = np.zeros((self.N_theta, self.N_x, self.K, self.T))
        for i in range(self.N_theta):
            for k in range(self.K):
                self.ancestors[i, :, k, 0] = np.arange(self.N_x)
        self.A = np.zeros((self.N_theta, self.K, self.K))
        self.Pi = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        self.Api = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        self.Phi = np.zeros((self.N_theta, self.K))


    def initialize_from_prior(self, y):

        # Sample A matrices
        for i in range(self.N_theta):
            for j in range(self.K):
                if j > 0:
                    self.A[i, j, :j] = np.random.multivariate_normal(self.mu_A[:j], self.Sigma_A[:j, :j])
                self.A[i, j, j] = 1.0

        # Sample Pi matrices
        for i in range(self.N_theta):
            for j in range(self.K):
                self.Pi[i, j, :] = np.random.multivariate_normal(
                    self.Pi_prior_mean[j], self.Pi_prior_var_rows[j]
                )

        # Compute Api
        for i in range(self.N_theta):
            self.Api[i] = self.A[i] @ self.Pi[i]

        # Sample Phi
        alpha = (self.K + 2) / 2
        beta = 1 / 2
        phi_samples = np.random.gamma(shape=alpha, scale=1.0, size=(self.N_theta, self.K))
        self.Phi = 1.0 / (phi_samples * beta)

        # Sample log-lambdas at time 0
        self.trajectories[:, :, :, 0] = np.random.normal(
            loc=0.0,
            scale= np.sqrt(self.sigma_0),
            size=(self.N_theta, self.N_x, self.K)
        )

    
    def intialize_from_posterior(self, y, N_burn_in, M, init_t):
        # note that in this initialization, log_lambdas at t = -1 (initialization from the prior)
        # is kept untouched, and therefore has value inf 
        
        print('start initialization')
        y_init = self.y[:init_t, :]
        print(y_init.shape)
        z_init = self.z[:init_t, :]
        gibbs_sampler  = GibbsSampler_API(
        y=y_init,
        z=z_init,
        mu_A=self.mu_A,
        Sigma_A=self.Sigma_A,
        Pi_prior_mean=self.Pi_prior_mean,
        Pi_prior_var=self.Pi_prior_var_rows,
        Pi_prior_var_inv=self.Pi_prior_var_rows_inv,
        phi = np.diag(self.Phi), # the values initialized from prior
        sigma0=self.sigma_0,
        T= init_t,
        N=self.K,
        p=self.p,
        #Num= self.N_x # so that we get the inner particle
        Num = 30,
        )  
        
        gibbs_sampler.run(N_burn_in + M*self.N_theta)
        
        for i in range(self.N_theta):
            self.A[i] = gibbs_sampler.samples['A'][N_burn_in + i*M]
            self.Pi[i] = gibbs_sampler.samples['Pi'][N_burn_in + i*M]
            self.Phi[i] = gibbs_sampler.samples['phi'][N_burn_in + i*M]
            # fixed particles
            log_lambda_i = gibbs_sampler.samples['log_lambdas'][N_burn_in + i*M]  # shape (1,K, T)
            # Precompute Api = A @ Pi (shape: K x (K*p + 1))
            Api = self.A[i] @ self.Pi[i]
            self.Api[i] = Api
            Ay_init = y_init @ self.A[i].T 

            # run CSMC to get the particle clouds
            for j in range(self.K):
                csmc = CSMC(
                    Num=self.N_x, # extend number of particles to N_x
                    phi=self.Phi[i],
                    sigma0=self.sigma_0,
                    y= Ay_init,
                    z=z_init,
                    B=Api,
                    j=j,
                    fixed_particles=log_lambda_i[j, :]
                )
                self.trajectories[i, :, j, 1:init_t + 1], self.ancestors[i, :, j, :init_t],_, _, _ = csmc.run(if_reconstruct=True)
                # log_lambdas has shape (, T+1)
                # but the chains passed from CSMC has shape (,T)

        print(f'finish intialization ')
    
    def intialize_from_posterior_result(self, y, results):
        self.init_prior(y)
        self.trajectories[:, :, :, 1:init_t + 1] = results['trajectories'] 
        self.ancestors[:, :, :, :init_t ] = results['ancestors']
        self.A = results['A']
        self.Pi = results['Pi']
        self.Api = np.matmul(self.A, self.Pi)
        self.Phi = results['Phi']
        #self.log_lambdas = jnp.full((self.N_theta, self.N_x, self.K, self.T + 1), jnp.inf)
        #self.log_lambdas = self.log_lambdas. at[:, :, :, 1:init_t + 1].set(results['log_lambdas'] )
        
        
    def resampling(self):
        # may not need safe resample if one remove potential NaN right after all rejuvenation
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)
        indices = np.searchsorted(np.cumsum(probs), np.random.rand(self.N_theta))
        
        self.A = self.A[indices]
        self.Pi = self.Pi[indices]
        self.Api = self.Api[indices]
        self.Phi = self.Phi[indices]
        self.trajectories = self.trajectories[indices]
        self.ancestors = self.ancestors[indices]
        self.outer_weights = np.zeros((self.N_theta))
        self.inner_weights = self.inner_weights[indices]


    def safe_replace_after_rejuvenation(self):
        # Define what counts as a "bad" particle
        is_bad = (
            np.any(np.isnan(self.A), axis=(1, 2)) |
            np.any(np.isnan(self.Pi), axis=(1, 2)) |
            np.any(np.isnan(self.Phi), axis=1) |
            np.any(np.isnan(self.trajectories), axis=(1, 2, 3))
        )

        num_bad = np.sum(is_bad)
        if num_bad == 0:
            return  # nothing to do

        self.log(f"[SafeGuard] Detected {int(num_bad)} NaN/Inf particles after rejuvenation — replacing.")

        # Prepare weights for resampling valid particles
        valid_mask = ~is_bad
        weights = self.outer_weights.copy()
        weights[~np.isfinite(weights)] = -np.inf
        weights[is_bad] = -np.inf

        max_weight = np.max(weights[valid_mask]) if np.any(valid_mask) else -np.inf
        stable_weights = np.exp(weights - max_weight)
        probs = stable_weights / np.sum(stable_weights)

        # Get indices of valid particles
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError("All particles are invalid — safe replacement not possible.")

        # Sample replacement indices from valid ones
        replacement_indices = np.random.choice(valid_indices, size=num_bad, p=probs[valid_indices])

        bad_indices = np.where(is_bad)[0]

        # Replace only bad particles
        self.A[bad_indices] = self.A[replacement_indices]
        self.Pi[bad_indices] = self.Pi[replacement_indices]
        self.Api[bad_indices] = self.Api[replacement_indices]
        self.Phi[bad_indices] = self.Phi[replacement_indices]
        self.trajectories[bad_indices] = self.trajectories[replacement_indices]
        self.ancestors[bad_indices] = self.ancestors[replacement_indices]
        self.inner_weights[bad_indices] = self.inner_weights[replacement_indices]
        self.outer_weights[bad_indices] = -np.inf  # Keep weight down until next proper resample


    def rejuvenate(self, rejuvenate_A = False):
        self.resampling()

        logsumexp_per_theta = scipy.special.logsumexp(self.inner_weights, axis=1, keepdims=True)
        logsumexp_per_theta = np.where(np.isfinite(logsumexp_per_theta), logsumexp_per_theta, 0.0)
        self.inner_weights -= logsumexp_per_theta
        weights = np.exp(self.inner_weights)
        
        
        for i in range(self.N_theta):
            # Select one fixed trajectory (log_lambda) for each coordinate k
            df_phi = []

            fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf)
            fix_log_lambda_phi_i = []
            for k in range(self.K):
                idx = np.random.choice(self.N_x, size=1, p=weights[i,:, k])
                if k < self.current_k + 1:  # observed coordinate
                # fix here, should not include the state from prior
                    df_phi.append(self.current_t + 1)
                    for t in reversed(range(self.current_t + 1)): # range from 0 to current_t:
                        fix_log_lambda_i[k,t] = self.trajectories[i,idx,k,t + 1]
                        idx = self.ancestors[i,idx,k,t].astype(int)
                    fix_log_lambda_phi_i.append(fix_log_lambda_i[k,:])
                else:
                    df_phi.append(self.current_t)
                    for t in reversed(range(self.current_t )): # range from 0 to current_t -1:
                        fix_log_lambda_i[k,t] = self.trajectories[i,idx,k,t + 1]
                        idx = self.ancestors[i,idx,k,t].astype(int)
                    fix_log_lambda_phi_i.append(fix_log_lambda_i[k,:self.current_t + 1 ])

            df_phi = np.array(df_phi)

            # Prepare required variables
            A_i = self.A[i]
            Pi_i = self.Pi[i]
            Phi_i = self.Phi[i]

            # Run one Gibbs step
            Ay = self.y @ self.A[i].T  # shape (T, K)
            for j in range(self.N_rejuvenate):
                Pi_i = update_APi(
                    Ay[:self.current_t + 1], self.z[:self.current_t + 1], Pi_i, fix_log_lambda_i,self.p,
                    self.A[i], self.Pi_prior_mean, self.Pi_prior_var_rows_inv
                )
            
                Phi_i = compute_posterior_phi_by_component(
                    fix_log_lambda_phi_i, df_phi, K=self.K
                )
    

                fix_log_lambda_phi_i = []
                trajectories = np.zeros((self.N_x, self.K, self.current_t + 1))
                ancestors = np.zeros((self.N_x, self.K, self.current_t + 1))
                for k in range(self.K):
                    if k < self.current_k + 1:
                        t_at_k = self.current_t + 1
                    else:
                        t_at_k = self.current_t 

                    csmc = CSMC(
                    Num=self.N_x,
                    phi=Phi_i,
                    sigma0=self.sigma_0,
                    y= Ay[:t_at_k],
                    z=self.z[:t_at_k],
                    B= np.matmul(A_i, Pi_i),
                    j=k,
                    fixed_particles= fix_log_lambda_i[k, :t_at_k],
                    )
                    trajectories[:,k,:t_at_k], ancestors[:,k,:t_at_k], sampled_trajectory ,_,_ = csmc.run(if_reconstruct=True)

                    fix_log_lambda_phi_i.append(sampled_trajectory)
                    fix_log_lambda_i[k, :t_at_k] = sampled_trajectory
                # the trejactories returned from CSMC are equally weighted


                if rejuvenate_A :
                    A_i = compute_posterior_A_with_log_lambdas(
                        self.y[:self.current_t + 1], self.z[:self.current_t +1], fix_log_lambda_i, self.mu_A, self.Sigma_A, Pi_i
                    )
                    Ay = (A_i @ self.y.T).T
            self.A[i] = A_i
            self.Pi[i] = Pi_i
            self.Api[i] = np.matmul(A_i, Pi_i)
            self.Phi[i] = Phi_i
            self.trajectories[i,:,:, 1: self.current_t + 2] = trajectories
            self.ancestors[i,:,:, : self.current_t + 1] = ancestors
            self.inner_weights[i] = np.zeros((self.N_x, self.K))
        
    
        self.log('[✓] Finished rejuvenation .')
        self.safe_replace_after_rejuvenation()



    def run(self, y, t_init = 0, init_result=None, if_pred = False):

        self.init_prior(y)

        self.ESS_track = []
        self.likelihood_track = []
        self.predicted_residuals = []

        if t_init == 0:
            self.initialize_from_prior(self.y)
        elif init_result is None:
            self.intialize_from_posterior(self.y, N_burn_in=1000, M=1, init_t=t_init)
        else:
            self.intialize_from_posterior_result(self.y, init_result)


        for t in range(t_init, self.T):
            self.current_t = t
            self.log(f'at time {t}')

            for k in range(self.K):
                self.current_k = k
                marginal_ll , ess_k = self.smc_step_single(t, k)
                self.ESS_track.append(ess_k)
                
                self.log(f'at time t = {t}, step = {k}, marginal log_likelihood = {marginal_ll}')
                self.likelihood_track.append(marginal_ll)
                self.log(f'ess at step {k} {ess_k}')

                if ess_k < self.threshold * self.N_theta and k < self.K - 1:
                    self.rejuvenate()
                elif ess_k < self.threshold_A * self.N_theta and k == self.K - 1: # and t > 10:
                    self.rejuvenate(rejuvenate_A = True)

            self.log(f'final ess {ess_k}')
            
            if if_pred:
                pred = self.monte_carlo_predictive_residuals()
                self.log(f'at time t = {t}, monte_carlo_predictive_residuals = {pred}')
                self.predicted_residuals.append(pred)

    def monte_carlo_predictive_residuals(self, num_samples=1000):
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)

        indices = np.random.choice(self.N_theta, size=num_samples, p=probs)
        predicted_residuals = []

        for i in range(num_samples):
            idx = indices[i]
            A_i = self.A[idx]
            A_i_inv = np.linalg.inv(A_i)
            Pi_i = self.Pi[idx]

            idxs = np.random.randint(0, self.N_x, size=self.K)
            fix_log_lambda_i = self.log_lambdas[idx, idxs, np.arange(self.K), self.current_t]
            Lambda_i = np.diag(np.exp(fix_log_lambda_i))
            # should be A^{-1} here 
            Sigma_i = A_i_inv @ Lambda_i @ A_i_inv.T

            try:
                residual_i = np.random.multivariate_normal(mean=np.zeros(self.K), cov=Sigma_i)
            except np.linalg.LinAlgError:
                self.log(f"[Warning] Non PD covariance at sample {i}, fallback to small noise.")
                residual_i = np.random.normal(0, 1e-4, size=self.K)

            predicted_residuals.append(residual_i)

        predicted_residuals = np.array(predicted_residuals)
        lower = np.percentile(predicted_residuals, 2.5, axis=0)
        upper = np.percentile(predicted_residuals, 97.5, axis=0)
        mean_residual = np.mean(predicted_residuals, axis=0)

        return np.array([lower, upper, mean_residual])

    def smc_step_single(self, t, k):

        # (N_theta, N_x) log volatilities at previous time
        log_lambda_prev_k = np.zeros((self.N_theta, self.N_x))
        for i in range(self.N_theta):
            if t == 0:
                log_lambda_prev_k[i] = self.trajectories[i,:, k, t]
            else:
                idx = self.ancestors[i,:,k,t - 1].astype(int)
                log_lambda_prev_k[i] = self.trajectories[i,idx, k, t]
        Phi_k = np.array(self.Phi[:, k]).reshape(-1, 1)
        noise = np.random.normal(size=(self.N_theta, self.N_x))
        # update trajectories
        log_lambda_t_k = log_lambda_prev_k + np.sqrt(Phi_k) * noise
        self.trajectories[:, :, k, t+1] = log_lambda_t_k

       # Safeguard log-lambda (log-volatility)
       #log_lambda_t_k = np.clip(log_lambda_t_k, a_min=-10.0, a_max=10.0)

        A_k = np.array(self.A[:, k, :]).reshape(self.N_theta, 1, self.K)
        Api_k = np.array(self.Api[:, k, :]).reshape(self.N_theta, 1, -1)
        z_t = np.array(self.z[t]).reshape(1, 1, -1)
        y_t = np.array(self.y[t]).reshape(1, 1, self.K)

        residuals = np.squeeze(
            np.matmul(A_k, y_t.transpose(0, 2, 1)) - np.matmul(Api_k, z_t.transpose(0, 2, 1)),
            axis=-1
        )
        # Clean residuals
        #residuals = np.nan_to_num(residuals, nan=0.0, posinf=1e6, neginf=-1e6)

        log_weights = -0.5 * (np.log(2 * np.pi) + log_lambda_t_k + np.exp(-log_lambda_t_k) * residuals**2)

        valid_mask = ~np.isnan(log_weights)
        if np.any(~valid_mask):
            self.log(f"[Warning] {np.sum(~valid_mask)} NaNs in log_weights at time t={t}, k={k} — replacing with -inf")
        safe_log_weights = np.where(valid_mask, log_weights, -np.inf)

        # Normalized log weights
        logsumexp_per_theta = scipy.special.logsumexp(self.inner_weights, axis=1, keepdims=True)
        logsumexp_per_theta = np.where(np.isfinite(logsumexp_per_theta), logsumexp_per_theta, 0.0)
        self.inner_weights -= logsumexp_per_theta

        ll_k = scipy.special.logsumexp(self.inner_weights[:,:,k] + safe_log_weights, axis=1, keepdims=True)

        # update inner weight
        self.inner_weights[:,:,k] += safe_log_weights
        # Normalized log weights again
        logsumexp_per_theta = scipy.special.logsumexp(self.inner_weights, axis=1, keepdims=True)
        logsumexp_per_theta = np.where(np.isfinite(logsumexp_per_theta), logsumexp_per_theta, 0.0)
        self.inner_weights -= logsumexp_per_theta
        weights = np.exp(self.inner_weights)
        #weights = np.clip(weights, 1e-12, 1.0)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = np.where(weights_sum > 0, weights / weights_sum, np.full_like(weights, 1.0 / self.N_x))

        # Resampling log_lambda_t_k using normalized weights
        for i in range(self.N_theta):
            weights_i = weights[i, :, k]
            inner_ESS = 1/np.sum(weights_i**2)
            if inner_ESS < self.threshold * self.N_x:
                idx = np.random.choice(self.N_x, size=self.N_x, p=weights_i)
                self.ancestors[i, :, k, t] = idx
                # reset inner weights
                self.inner_weights[i, :, k] = np.zeros(self.N_x)
            else:
                self.ancestors[i, :, k, t] = np.arange(self.N_x)

        # compute marginal likelihood
        outer_norm = self.outer_weights - scipy.special.logsumexp(self.outer_weights)
        marginal_ll = scipy.special.logsumexp(outer_norm + np.squeeze(ll_k))
        
        # update outer weight
        self.outer_weights += np.squeeze(ll_k)
        # Normalize and compute ESS
        self.outer_weights -= np.max(self.outer_weights)
        outer_weight = np.exp(self.outer_weights)
        outer_weight /= np.sum(outer_weight)
        ess_k = 1.0 / np.sum(outer_weight**2)

        return marginal_ll, ess_k
