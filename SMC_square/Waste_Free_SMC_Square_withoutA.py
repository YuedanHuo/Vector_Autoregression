import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import scipy
import time
import scipy.special
import os
import threading
from collections import defaultdict
from multiprocessing import Manager

sys.path.append('/files/Gibbs_sampler')
sys.path.append('/files/experiment')
from prior_setup import BayesianVARPrior
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi, compute_posterior_phi_by_component
from SMC import SMC
sys.path.append('/files/Gibbs_sampler/CSMC_gibbs')
from CSMC_bs import CSMC
sys.path.append('/files/Gibbs_sampler/APi_Gibbs')
from Update_APi import update_APi
from Gibbs_Api import GibbsSampler_API
sys.path.append('/files/SMC_square')
from Initialization import initialize_from_posterior




class WF_SMC_Square:
    def __init__(self, p, logfile, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.3):
        self.p = p
        self.N_theta = N_theta
        self.N_x = N_x
        self.threshold = threshold
        self.N_rejuvenate = N_rejuvenate
        self.logfile = logfile
        self.need_rejuvenate_A = False
        self.M_theta = N_theta // (N_rejuvenate +1) # for waste-free-smc

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
        print(self.T)

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
    
    def intialize_from_posterior_result(self, results, init_t):
        
        for i in range(self.M_theta):
            sample_result = results[i]
            for j in range(self.N_rejuvenate + 1):
                target_index = i * (self.N_rejuvenate+1) + j
                self.A[target_index] = sample_result['A'][j]
                self.Pi[target_index] = sample_result['Pi'][j]
                self.Phi[target_index] = sample_result['Phi'][j]
                self.Api[target_index] = self.A[target_index] @ self.Pi[target_index]
            self.trajectories[i*(self.N_rejuvenate + 1) : (i+1)*(self.N_rejuvenate + 1),:,:, 0:init_t + 1] = sample_result['trajectories']
            self.ancestors[i*(self.N_rejuvenate + 1) : (i+1)*(self.N_rejuvenate + 1),:,:, 0:init_t] = sample_result['ancestors']
    


    def waste_free_resampling(self):
        # may not need safe resample if one remove potential NaN right after all rejuvenation
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)
        indices = np.searchsorted(np.cumsum(probs), np.random.rand(self.M_theta))
        
        self.A[:self.M_theta] = self.A[indices]
        self.Pi[:self.M_theta] = self.Pi[indices]
        self.Api[:self.M_theta] = self.Api[indices]
        self.Phi[:self.M_theta] = self.Phi[indices]
        self.trajectories[:self.M_theta] = self.trajectories[indices]
        self.ancestors[:self.M_theta] = self.ancestors[indices]
        # renew all outer weights
        self.outer_weights = np.zeros(self.N_theta)
        self.inner_weights[:self.M_theta] = self.inner_weights[indices]


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

    
    def _rejuvenate_one_theta(self, i, rejuvenate_A, tracker):

        #add tracking of resources
        if tracker is not None:
            tracker.append(threading.get_ident())  


        logw = self.inner_weights[i]
        logw -= scipy.special.logsumexp(logw, axis=0, keepdims=True)
        weights = np.exp(logw)

        fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf)
        fix_log_lambda_phi_i = []
        df_phi = []

        for k in range(self.K):
            idx = np.random.choice(self.N_x, p=weights[:, k])
            if k <= self.current_k:
                df_phi.append(self.current_t + 1)
                for t in reversed(range(self.current_t + 1)):
                    fix_log_lambda_i[k, t] = self.trajectories[i, idx, k, t + 1] # does not include the initialization
                    idx = self.ancestors[i, idx, k, t].astype(int)
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :])
            else:
                df_phi.append(self.current_t)
                for t in reversed(range(self.current_t)):
                    fix_log_lambda_i[k, t] = self.trajectories[i, idx, k, t + 1]
                    idx = self.ancestors[i, idx, k, t].astype(int)
                # correct bug here
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :self.current_t + 1])

        A_i = self.A[i]
        Pi_i = self.Pi[i]
        Phi_i = self.Phi[i]
        Ay = self.y @ A_i.T

        for j in range(self.N_rejuvenate):
            Pi_i = update_APi(
                Ay[:self.current_t + 1], self.z[:self.current_t + 1], Pi_i, fix_log_lambda_i, self.p,
                A_i, self.Pi_prior_mean, self.Pi_prior_var_rows_inv
            )
            Phi_i = compute_posterior_phi_by_component(
                fix_log_lambda_phi_i, df_phi, K=self.K
            )
            Api_i = A_i @ Pi_i

            # set the target index
            target_idx = self.M_theta + i * self.N_rejuvenate + j

            fix_log_lambda_phi_i = []
            for k in range(self.K):
                t_k = self.current_t + 1 if k <= self.current_k else self.current_t
                csmc = CSMC(
                Num=self.N_x,
                phi=Phi_i,
                sigma0=self.sigma_0,
                y=Ay[:t_k],
                z=self.z[:t_k],
                B=Api_i,
                j=k,
                fixed_particles=fix_log_lambda_i[k, :t_k]
                )
                self.trajectories[target_idx, :, k, 1:t_k + 1], self.ancestors[target_idx, :, k, :t_k], sampled_traj, _, _ = csmc.run(if_reconstruct=True)
                
                fix_log_lambda_phi_i.append(sampled_traj)
                fix_log_lambda_i[k, :t_k] = sampled_traj
            
            # add the new rejuvenation of A
            #APiz = A_i @ Pi_i @ self.z[self.current_t]
            #for k in range(self.current_k +1, self.K):
                #for k > self.current_k:
                # sample future volatility
                ##### correct the bug here!
                #fix_log_lambda_i[k, self.current_t] = fix_log_lambda_i[k, self.current_t -1] + np.sqrt(Phi_i[k]) * np.random.normal(0,1)
                # sampling unobserved tilde{y}, replace in placed in Ay
                #Ay[self.current_t, k] = np.random.normal(
                #    APiz[k], 
                    #### correct the bug here!
                #    np.exp(fix_log_lambda_i[k, self.current_t]) ** .5
                #)

            #syn_y = Ay[:self.current_t + 1] @ np.linalg.inv(A_i.T)
            if rejuvenate_A == True:
                A_i = compute_posterior_A_with_log_lambdas(
                    self.y[:self.current_t + 1], self.z[:self.current_t + 1],
                    fix_log_lambda_i, self.mu_A, self.Sigma_A, Pi_i
                )
            
            self.A[target_idx] = A_i
            self.Pi[target_idx] = Pi_i
            self.Phi[target_idx] = Phi_i
            self.Api[target_idx] = Api_i
            self.inner_weights[target_idx] = np.zeros((self.N_x, self.K))
                
    
    def rejuvenate_parallelized(self, rejuvenate_A=False):
        self.waste_free_resampling()
        
        # normalized inner weights for later passed-in
        logsumexp_per_theta = scipy.special.logsumexp(self.inner_weights, axis=1, keepdims=True)
        logsumexp_per_theta = np.where(np.isfinite(logsumexp_per_theta), logsumexp_per_theta, 0.0)
        self.inner_weights -= logsumexp_per_theta

        with Manager() as manager:
            tracker = manager.list()
            Parallel(n_jobs=-1, backend="threading")(
                delayed(self._rejuvenate_one_theta)(i, rejuvenate_A, tracker)
                for i in range(self.M_theta)
            )

            unique_workers = set(tracker)
            self.log(f"[✓] Rejuvenation complete. Used {len(unique_workers)} parallel workers.")
        self.safe_replace_after_rejuvenation()


    def run(self, y, t_init=0, init_result=None, if_pred=False):

        self.init_prior(y)

        self.ESS_track = []
        self.likelihood_track = []
        self.predicted_residuals = []

        if t_init == 0:
            self.initialize_from_prior(self.y)
        else:
            if init_result is None:
                init_result = initialize_from_posterior(y, self.p, t_init, self.N_rejuvenate, self.N_x, self.M_theta)
            self.intialize_from_posterior_result(init_result, t_init)
        

        for t in range(t_init, self.T):
            self.current_t = t
            self.log(f'at time {t}')

            for k in range(self.K):
                self.current_k = k
                marginal_ll, ess_k = self.smc_step_single_parallel(t, k)
                self.ESS_track.append(ess_k)
                
                self.log(f'at time t = {t}, step = {k}, marginal log_likelihood = {marginal_ll}')
                self.likelihood_track.append(marginal_ll)
                self.log(f'ess at step {k} {ess_k}')

                if ess_k < self.threshold * self.N_theta and k < self.K:
                    self.rejuvenate_parallelized()
                elif ess_k < 0.6 * self.N_theta and k == self.K:
                    self.rejuvenate_parallelized(rejuvenate_A= True)

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

    
    def _smc_inner_loop(self, i, t, k, Phi_k, A_k, Api_k, y_t, z_t):
        # single step for the ith theta particle
        if t == 0:
            log_lambda_prev = self.trajectories[i, :, k, t]
        else:
            idx = self.ancestors[i, :, k, t-1].astype(int)
            log_lambda_prev = self.trajectories[i, idx, k, t]

        log_lambda_t = log_lambda_prev + np.sqrt(Phi_k[i]) * np.random.normal(size=self.N_x)
        log_lambda_t = np.clip(log_lambda_t, a_min=-10.0, a_max=10.0)
        self.trajectories[i, :, k, t + 1] = log_lambda_t

        A = A_k[i].reshape(1, -1) 
        Api = Api_k[i].reshape(1, -1)
        res = (A @ y_t - Api @ z_t).squeeze()
        res = np.clip(res, -1e3, 1e3)

        lw = -0.5 * (np.log(2 * np.pi) + log_lambda_t + np.exp(-log_lambda_t) * res**2)
        lw = np.where(np.isnan(lw), -np.inf, lw)

        self.inner_weights[i, :, k] += lw
        #ll_i = scipy.special.logsumexp(self.inner_weights[i, :, k]) - np.log(self.N_x)
        ll_i = scipy.special.logsumexp(lw) - np.log(self.N_x)

        #log_w = self.inner_weights[i, :, k] - scipy.special.logsumexp(self.inner_weights[i, :, k])
        self.inner_weights[i, :, k] -= scipy.special.logsumexp(self.inner_weights[i, :, k])
        weights = np.exp(self.inner_weights[i, :, k])
        weights /= np.sum(weights) if np.sum(weights) > 0 else self.N_x

        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.threshold * self.N_x:
            new_idx = np.random.choice(self.N_x, size=self.N_x, p=weights)
            self.ancestors[i, :, k, t] = new_idx
            self.inner_weights[i, :, k] = np.zeros(self.N_x)
        else:
            self.ancestors[i, :, k, t] = np.arange(self.N_x)
        
        return ll_i
    
    def smc_step_single_parallel(self, t, k):

        Phi_k = np.array(self.Phi[:, k]).reshape(-1, 1)
        A_k = np.array(self.A[:, k, :]).reshape(self.N_theta, 1, self.K)
        Api_k = np.array(self.Api[:, k, :]).reshape(self.N_theta, 1, -1)
        #z_t = np.array(self.z[t]).reshape(1, 1, -1)
        #y_t = np.array(self.y[t]).reshape(1, 1, self.K)
        y_t_vec = self.y[t].reshape(-1, 1)   # shape: (K, 1)
        z_t_vec = self.z[t].reshape(-1, 1)   # shape: (D, 1)

        ll_k = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._smc_inner_loop)(
                i, t, k, Phi_k, A_k, Api_k, y_t_vec, z_t_vec
                )
            for i in range(self.N_theta)
            )

        ll_k = np.array(ll_k).reshape(-1, 1)


        # compute marginal likelihood
        outer_norm = self.outer_weights - scipy.special.logsumexp(self.outer_weights)
        marginal_ll = scipy.special.logsumexp(outer_norm + np.squeeze(ll_k))
        if marginal_ll > 0:
            self.log(f"[Warning] marginal_ll = {marginal_ll:.4f} > 0 at t={t}, likely numerical issue. Max ll_k: {np.max(ll_k)}, max outer weight: {np.max(outer_norm)}")

        # update outer weight
        self.outer_weights += np.squeeze(ll_k)
        # Normalize and compute ESS
        self.outer_weights -= np.max(self.outer_weights)
        outer_weight = np.exp(self.outer_weights)
        outer_weight /= np.sum(outer_weight)
        ess_k = 1.0 / np.sum(outer_weight**2)

        return marginal_ll, ess_k

