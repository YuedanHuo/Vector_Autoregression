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

sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from prior_setup import BayesianVARPrior
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi, compute_posterior_phi_by_component
from SMC import SMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/CSMC_gibbs')
from CSMC_bs import CSMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/APi_Gibbs')
from Update_APi import update_APi
from Gibbs_Api import GibbsSampler_API
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/SMC_square')
from Initialization import initialize_from_posterior
from ParticleFilter_Jacob import AncestryNode, ParticleFilterJacob

# for debugging
import gc
import weakref
node_registry = weakref.WeakSet()
import tracemalloc
tracemalloc.start()


class WF_SMC_Square_Jacob:
    def __init__(self, p, logfile, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.4):
        self.p = p
        self.N_theta = N_theta
        self.N_x = N_x
        self.threshold = threshold
        self.N_rejuvenate = N_rejuvenate
        self.logfile = logfile
        self.need_rejuvenate_A = False
        self.M_theta = N_theta // (N_rejuvenate +1) # for waste-free-smc

        self.node_counts = [] # for debugging
        self.mem_usage = []

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
        self.outer_particles = []

        for i in range(self.N_theta):
            pf_instances_for_this_theta =[]
            for k in range(self.K):
                initial_theta_i_k  = {
                    'A': np.zeros((1, self.K)),
                    'Api': np.zeros((1, self.K * self.p + 1)),
                    'Phi' : np.zeros(1)
                }
                # initial PF for each theta, each dimension
                pf_i_k = ParticleFilterJacob(
                    N_x=self.N_x,
                    T=self.T,
                    state_dim=1,
                    #model_functions=self.model_functions,
                    theta=initial_theta_i_k # The theta for THIS parameter particle
                )
                pf_i_k.initialize(node_registry)
                pf_instances_for_this_theta.append(pf_i_k)
            self.outer_particles.append({
                #'theta': initial_theta_i,
                'particle_filters': pf_instances_for_this_theta,
                'outer_weight': 0.0 # Will be updated after all PFs are run and initial likelihoods computed
            })
    
    ################# ignore this function for now ##############
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
                A = sample_result['A'][j]
                Pi = sample_result['Pi'][j]
                Phi = sample_result['Phi'][j]
                Api = A @ Pi
                for k in range(self.K):
                    pf = self.outer_particles[target_index]['particle_filters'][k]
                    pf.theta = {
                    'A': A[k],
                    'Api': Api[k],
                    'Phi' : Phi[k]
                    }
                    pf.load_from_matrix_form(sample_result['trajectories'][j,:,k, 1:init_t + 1, np.newaxis], 
                                             sample_result['ancestors'][j,:,k,:init_t], node_registry)
            
  
    def waste_free_resampling(self):
        # may not need safe resample if one remove potential NaN right after all rejuvenation
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)
        indices = np.searchsorted(np.cumsum(probs), np.random.rand(self.M_theta))

        import copy
        new_parameter_particles = []
        for i in range(self.M_theta):
            #outer_particles_i = self.outer_particles[indices[i]]
            outer_particles_i = copy.deepcopy(self.outer_particles[indices[i]])
            outer_particles_i['log_weight'] = 0.0
            new_parameter_particles.append(outer_particles_i)
        
        self.outer_particles[:self.M_theta] = new_parameter_particles

        # Clear ancestry in all particles beyond M_theta (assumed inactive)
        for i in range(self.M_theta, self.N_theta):
            for k in range(self.K):
                self.outer_particles[i]['particle_filters'][k].reset_ancestry()
        
        # force gc collection here
        # can delete later
        gc.collect()
        #self.log(f"Waste-free resampling: Explicitly cleared ancestry for {cleaned_up_count} unselected particles.")
        # renew all outer weights
        self.outer_weights = np.zeros(self.N_theta)
    

######## ignore this for now ##############
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

    
    def _rejuvenate_one_theta(self, i, tracker):

        #add tracking of resources
        if tracker is not None:
            tracker.append(threading.get_ident())  

        fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf)
        fix_log_lambda_phi_i = []
        df_phi = []
        
        pf_i = self.outer_particles[i]['particle_filters']

        ######### here get the path from PF objects ########
        for k in range(self.K):
            path = pf_i[k].sample_trajectory()
            if k <= self.current_k:
                df_phi.append(self.current_t + 1)
                fix_log_lambda_i[k, :] = path 
                fix_log_lambda_phi_i.append(path)
            else:
                df_phi.append(self.current_t)
                fix_log_lambda_i[k, :self.current_t] = path
                fix_log_lambda_phi_i.append(path)
        
        A_i = np.zeros((self.K,self.K))
        APi_i = np.zeros((self.K, self.K * self.p + 1))
        Phi_i = np.zeros(self.K)
        for k in range(self.K):
            A_i[k] = self.outer_particles[i]['particle_filters'][k].theta['A']
            APi_i[k] = self.outer_particles[i]['particle_filters'][k].theta['Api']
            Phi_i[k] = self.outer_particles[i]['particle_filters'][k].theta['Phi']
        #Pi_i = np.linalg.inv(A_i) @ APi_i
        Pi_i = np.linalg.solve(A_i, APi_i)
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
            new_trajectories = []
            new_ancestors = []
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
                trajectories, ancestors, sampled_traj, _, _ = csmc.run(if_reconstruct=True)
                # Explicitly cut off old references
                #self.outer_particles[target_idx]['particle_filters'][k].reset_ancestry()
                #self.outer_particles[target_idx]['particle_filters'][k].load_from_matrix_form(trajectories[:,:,np.newaxis], ancestors, node_registry)
                new_trajectories.append(trajectories[:,:,np.newaxis])
                new_ancestors.append(ancestors)
                fix_log_lambda_phi_i.append(sampled_traj)
                fix_log_lambda_i[k, :t_k] = sampled_traj

            for k in range(self.K):
                # Explicitly cut off old references FIRST
                self.outer_particles[target_idx]['particle_filters'][k].reset_ancestry()
                # can force gc collection
                #gc.collect() # Use with caution, can slow down

                # Then load the new form, creating new AncestryNode objects
                self.outer_particles[target_idx]['particle_filters'][k].load_from_matrix_form(
                    new_trajectories[k], new_ancestors[k], node_registry
                )
            

            # add the new rejuvenation of A
            APiz = A_i @ Pi_i @ self.z[self.current_t]
            for k in range(self.current_k +1, self.K):
                #for k > self.current_k:
                # sample future volatility
                ##### correct the bug here!
                fix_log_lambda_i[k, self.current_t] = fix_log_lambda_i[k, self.current_t -1] + np.sqrt(Phi_i[k]) * np.random.normal(0,1)
                # sampling unobserved tilde{y}, replace in placed in Ay
                Ay[self.current_t, k] = np.random.normal(
                    APiz[k], 
                    #### correct the bug here!
                    np.exp(fix_log_lambda_i[k, self.current_t]) ** .5
                )

            syn_y = Ay[:self.current_t + 1] @ np.linalg.inv(A_i.T)
            A_i = compute_posterior_A_with_log_lambdas(
                syn_y, self.z[:self.current_t + 1],
                fix_log_lambda_i, self.mu_A, self.Sigma_A, Pi_i
            )
            
            for k in range(self.K):
                self.outer_particles[target_idx]['particle_filters'][k].theta = {
                    'A': A_i[k],
                    'Api': Api_i[k],
                    'Phi': Phi_i[k]
                }


    def rejuvenate_parallelized(self):
        self.waste_free_resampling()
        

        with Manager() as manager:
            tracker = manager.list()
            Parallel(n_jobs=-1, backend="threading")(
                delayed(self._rejuvenate_one_theta)(i, tracker)
                for i in range(self.M_theta)
            )

            unique_workers = set(tracker)
            self.log(f"[✓] Rejuvenation complete. Used {len(unique_workers)} parallel workers.")
        #self.safe_replace_after_rejuvenation()

    
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
                marginal_ll, ess_k = self.smc_step(t, k)
                self.ESS_track.append(ess_k)
                
                self.log(f'at time t = {t}, step = {k}, marginal log_likelihood = {marginal_ll}')
                self.likelihood_track.append(marginal_ll)
                self.log(f'ess at step {k} {ess_k}')
    
                if ess_k < self.threshold * self.N_theta:
                    self.rejuvenate_parallelized()
                
                # force garbage collection
                # can delete later
               # gc.collect() 
                node_count = len(node_registry)
                self.log(f'at time t = {t}, step = {k}, node_count = {node_count}')
                # for debugging
                self.node_counts.append(node_count)
                current, peak = tracemalloc.get_traced_memory()
                self.mem_usage.append(current / 1e6)


            self.log(f'final ess {ess_k}')
            
            if if_pred:
                pred = self.monte_carlo_predictive_residuals()
                self.log(f'at time t = {t}, monte_carlo_predictive_residuals = {pred}')
                self.predicted_residuals.append(pred)

################# ignore this function for now ##############
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

    def smc_step(self, t, k):
        ll_k = np.zeros(self.N_theta)
        # seems like it is even slower in parallelizing
        #def step_and_loglik(i):
        #    pf_i = self.outer_particles[i]['particle_filters']
        #    pf_i[k].step(t, self.y[t], self.z[t], node_registry)
        #    return pf_i[k].get_marginal_loglik()

        #ll_k = Parallel(n_jobs=-1,)(delayed(step_and_loglik)(i) for i in range(self.N_theta))
        #ll_k = np.array(ll_k)
        #print('shape of llk', ll_k.shape)
        for i in range(self.N_theta):
            pf_i = self.outer_particles[i]['particle_filters']
            pf_i[k].step(t, self.y[t],self.z[t], node_registry)
            ll_k[i] = pf_i[k].get_marginal_loglik()

        ll_k = np.array(ll_k)#.reshape(-1, 1)
        print('ll_k',ll_k)

        # compute marginal likelihood
        outer_norm = self.outer_weights - scipy.special.logsumexp(self.outer_weights)
        marginal_ll = scipy.special.logsumexp(outer_norm + ll_k)
        if marginal_ll > 0:
            self.log(f"[Warning] marginal_ll = {marginal_ll:.4f} > 0 at t={t}, likely numerical issue. Max ll_k: {np.max(ll_k)}, max outer weight: {np.max(outer_norm)}")

        # update outer weight
        self.outer_weights += ll_k
        # Normalize and compute ESS
        self.outer_weights -= np.max(self.outer_weights)
        outer_weight = np.exp(self.outer_weights)
        outer_weight /= np.sum(outer_weight)
        ess_k = 1.0 / np.sum(outer_weight**2)

        return marginal_ll, ess_k




