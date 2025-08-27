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
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/experiment')
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


import zarr
import dask.array as da
import os
import shutil # For removing directories
from dask.distributed import Client, LocalCluster, as_completed

# Adjust n_workers and memory_limit based on your Nuvolos instance's resources
# threads_per_worker=1 is often good for heavily NumPy-bound tasks to avoid GIL contention
#cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, memory_limit='auto') 
#client = Client(cluster)
#print(client) # Provides a link to the Dask dashboard - essential for monitoring!


class WF_SMC_Square:
    def __init__(self, p, y, logfile, zarr_path, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.4):
        self.p = p
        self.N_theta = N_theta
        self.N_x = N_x
        self.threshold = threshold
        self.N_rejuvenate = N_rejuvenate
        self.logfile = logfile
        self.need_rejuvenate_A = False
        self.M_theta = N_theta // (N_rejuvenate + 1) # for waste-free-smc

        self.init_prior(y)

        # Call the Zarr setup function
        self._setup_zarr_stores(zarr_path, overwrite_existing=True)

    def log(self, message):
        with open(self.logfile, 'a') as f:
            f.write(f"[{threading.get_ident()}] {message}\n")

    def init_prior(self, y):
        bayesian_var_prior = BayesianVARPrior(y, p=self.p, sigma_0=2.31, threshold=0.5)
        priors = bayesian_var_prior.get_priors()

        self.Sigma_A = priors["Sigma_A"]
        self.mu_A = priors["mu_A"]
        self.Pi_prior_mean = priors["Pi_prior_mean"]
        self.Pi_prior_var_rows_inv = priors["Pi_prior_var_inv"]
        self.sigma_0 = priors["sigma_0"]
        self.z = priors["Z"]
        self.y = priors["y"]
        self.T, self.K = self.y.shape
        print(f"Data dimensions: T={self.T}, K={self.K}")

        self.outer_weights = np.zeros((self.N_theta,))
        self.inner_weights = np.zeros((self.N_theta, self.N_x, self.K))
        self.A = np.zeros((self.N_theta, self.K, self.K))
        self.Pi = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        self.Api = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        self.Phi = np.zeros((self.N_theta, self.K))


    def _setup_zarr_stores(self, zarr_base_path, overwrite_existing=True):
        """
        Sets up Zarr arrays with one chunk per particle (N_theta chunks total).
        This is much more efficient for Dask than chunking every single element.
        """
        self.zarr_base_path = zarr_base_path
        if overwrite_existing and os.path.exists(self.zarr_base_path):
            self.log(f"Removing existing Zarr store at {self.zarr_base_path}")
            shutil.rmtree(self.zarr_base_path)
            
        root_group = zarr.open_group(self.zarr_base_path, mode='a')
        self.log(f"Setting up Zarr stores at {self.zarr_base_path}")

        # --- CORRECTED CHUNKING STRATEGY ---
        # We create exactly N_theta chunks. Each chunk contains all data for one particle.
        # Shape of each chunk will be (1, N_x, K, T+1)
        
        # 1. trajectories: (N_theta, N_x, K, T+1)
        self.store_trajectories = root_group.zeros(
            'trajectories', 
            shape=(self.N_theta, self.N_x, self.K, self.T + 1), 
            dtype='float64', 
            chunks=(1, self.N_x, self.K, self.T + 1) # One chunk per particle
        )
        self.trajectories = da.from_zarr(self.store_trajectories)

        # 2. ancestors: (N_theta, N_x, K, T)
        self.store_ancestors = root_group.zeros(
            'ancestors', 
            shape=(self.N_theta, self.N_x, self.K, self.T), 
            dtype='int64', 
            chunks=(1, self.N_x, self.K, self.T) # One chunk per particle
        )
        self.ancestors = da.from_zarr(self.store_ancestors)

        self.log(f"Zarr stores set up successfully with {self.trajectories.npartitions} partitions (chunks).")

    # ... The rest of your class methods remain the same ...
    # The changes to chunking do not require changes to the logic,
    # but they will significantly improve the performance of Dask operations.

    def intialize_from_posterior_result(self, results, init_t, client: Client):
        
        for i in range(self.M_theta):
            sample_result = results[i]
            for j in range(self.N_rejuvenate + 1):
                target_index = i * (self.N_rejuvenate+1) + j
                self.A[target_index] = sample_result['A'][j]
                self.Pi[target_index] = sample_result['Pi'][j]
                self.Phi[target_index] = sample_result['Phi'][j]
                self.Api[target_index] = self.A[target_index] @ self.Pi[target_index]
        # Define the destination slice for clarity
        destination_slice = slice(i * (self.N_rejuvenate + 1), (i + 1) * (self.N_rejuvenate + 1))

        self.store_trajectories[destination_slice, :, :, 0:init_t + 1] = sample_result['trajectories']
        self.store_ancestors[destination_slice, :, :, 0:init_t] = sample_result['ancestors']
    

    def waste_free_resampling(self, client: Client):
        """
        Performs waste-free resampling of particles.
        """
        self.log("Performing waste-free resampling...")
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)
        indices = np.searchsorted(np.cumsum(probs), np.random.rand(self.M_theta))
        
        self.A[:self.M_theta] = self.A[indices]
        self.Pi[:self.M_theta] = self.Pi[indices]
        self.Api[:self.M_theta] = self.Api[indices]
        self.Phi[:self.M_theta] = self.Phi[indices]
        self.inner_weights[:self.M_theta] = self.inner_weights[indices]
        self.outer_weights = np.zeros((self.N_theta,))

        self.log("Copying resampled particle data directly in Zarr...")
        # With the new chunking, this operation becomes much more efficient.
        # Dask will copy N_theta large chunks instead of millions of small ones.
        source_trajectories = self.trajectories[indices, :, :, :]
        source_ancestors = self.ancestors[indices, :, :, :]

        da.store([source_trajectories, source_ancestors],
                 [self.store_trajectories[:self.M_theta], self.store_ancestors[:self.M_theta]],
                 compute=True, scheduler='threads') # Use Dask's optimized store
    
        self.log("Waste-free resampling complete.")


    def _rejuvenate_one_theta(self, i, rejuvenate_A, tracker,
                                current_theta_trajectories_np, # ith particle (N_x, K, T+1)
                                current_theta_ancestors_np   # (N_x, K, T)
                                ):
        """
        Performs rejuvenation for a single theta particle.
        This function is designed to be called in parallel (e.g., via joblib or Dask client.map).
        It takes NumPy copies of relevant data and returns new Dask array chunks
        and updated NumPy parameter arrays.
        """
        #add tracking of resources
        if tracker is not None:
            tracker.append(threading.get_ident())  

  
        logw = self.inner_weights[i]
        logw -= scipy.special.logsumexp(logw, axis=0, keepdims=True)
        weights = np.exp(logw)

        fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf) # Use self.T+1 for max possible length
        fix_log_lambda_phi_i = []
        df_phi = []

        for k in range(self.K):
            idx = np.random.choice(self.N_x, p=weights[:, k])
            if k <= self.current_k:
                df_phi.append(self.current_t + 1)
                for t in reversed(range(self.current_t + 1)):
                    fix_log_lambda_i[k, t] = current_theta_trajectories_np[idx, k, t + 1] # does not include the initialization
                    idx = current_theta_ancestors_np[idx, k, t].astype(int)
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :self.current_t + 1]) # Slice up to current_t+1
            else:
                df_phi.append(self.current_t)
                for t in reversed(range(self.current_t)):
                    fix_log_lambda_i[k, t] = current_theta_trajectories_np[idx, k, t + 1]
                    idx = current_theta_ancestors_np[idx, k, t].astype(int)
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :self.current_t]) # Slice up to current_t

        A_i = self.A[i]
        Pi_i = self.Pi[i]
        Phi_i = self.Phi[i]
        Ay = self.y @ A_i.T

        # These lists will store the results for each of the N_rejuvenate steps for *this* particle
        all_trj_da = [] # List of Dask array chunks, each for one rejuvenation step
        all_ans_da = [] # List of Dask array chunks, each for one rejuvenation step

        for j in range(self.N_rejuvenate):
            Pi_i = update_APi(
                Ay[:self.current_t + 1], self.z[:self.current_t + 1], Pi_i, fix_log_lambda_i, self.p,
                A_i, self.Pi_prior_mean, self.Pi_prior_var_rows_inv
            )
            Phi_i = compute_posterior_phi_by_component(
                fix_log_lambda_phi_i, df_phi, K=self.K
            )
            Api_i = A_i @ Pi_i

            # --- CSMC and new trajectory generation for this rejuvenation step ---
            # These will hold the combined (N_x, K, ...) NumPy arrays for *one* rejuvenation step (j)
            concatenated_trj_for_j_np = np.zeros((self.N_x, self.K, self.T + 1))
            concatenated_ans_for_j_np = np.zeros((self.N_x, self.K, self.T), dtype=int)
            
            # The inner weights for this *newly generated particle* (start at zero for log-weights)
            #inner_weights_for_j_np = np.zeros((self.N_x, self.K), dtype='float64')

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
                fixed_particles=fix_log_lambda_i[k, :t_k] # Use current temp_fix
                )
                trj_k_np, ans_k_np, sampled_traj_k, _, _ = csmc.run(if_reconstruct=True) # Returns NumPy
                
                # Assign to the correct slices of the pre-allocated arrays
                concatenated_trj_for_j_np[:, k, 1:t_k + 1] = trj_k_np
                concatenated_ans_for_j_np[:, k, 0:t_k] = ans_k_np
                
                fix_log_lambda_phi_i.append(sampled_traj_k)
                fix_log_lambda_i[k, :t_k] = sampled_traj_k # Update copy for next k

            # add the new rejuvenation of A
            APiz = A_i @ Pi_i @ self.z[self.current_t]
            for k in range(self.current_k +1, self.K):
                #for k > self.current_k:
                # sample future volatility
                fix_log_lambda_i[k, self.current_t] = fix_log_lambda_i[k, self.current_t -1] + np.sqrt(Phi_i[k]) * np.random.normal(0,1)
                # sampling unobserved tilde{y}, replace in placed in Ay
                Ay[self.current_t, k] = np.random.normal(
                    APiz[k], 
                    np.exp(fix_log_lambda_i[k, self.current_t]) ** .5
                )

            syn_y = Ay[:self.current_t + 1] @ np.linalg.inv(A_i.T)
            A_i = compute_posterior_A_with_log_lambdas(
                syn_y, self.z[:self.current_t + 1],
                fix_log_lambda_i, self.mu_A, self.Sigma_A, Pi_i
            )
            
            # Store the NumPy arrays as Dask array chunks, each representing one new particle (1, N_x, K, T/T+1)
            #all_trj_da.append(da.from_array(concatenated_trj_for_j_np[:,:,:], 
             #                                          chunks = (self.N_x, self.K, self.T + 1)))
            #all_ans_da.append(da.from_array(concatenated_ans_for_j_np[:,:,:], 
            #                                           chunks = (self.N_x, self.K, self.T)))
            
            # set the target index
            target_idx = self.M_theta + i * self.N_rejuvenate + j
            self.A[target_idx] = A_i
            self.Pi[target_idx] = Pi_i
            self.Phi[target_idx] = Phi_i
            self.Api[target_idx] = Api_i
            self.inner_weights[target_idx] = np.zeros((self.N_x, self.K))
            self.store_trajectories[target_idx] = concatenated_trj_for_j_np
            self.store_ancestors[target_idx] = concatenated_ans_for_j_np

        # Return all collected Dask arrays and NumPy parameters
        #return (all_trj_da, all_ans_da)
    

    def rejuvenate_parallelized(self, client: Client, rejuvenate_A=False):
        """
        Coordinates parallel rejuvenation of all theta particles.
        """
        self.log("Starting parallel rejuvenation...")
        self.waste_free_resampling(client)
        
        logsumexp_per_theta = scipy.special.logsumexp(self.inner_weights, axis=1, keepdims=True)
        logsumexp_per_theta = np.where(np.isfinite(logsumexp_per_theta), logsumexp_per_theta, 0.0)
        self.inner_weights -= logsumexp_per_theta

        # With the new chunking, computing the first M_theta particles is efficient.
        self.log("Computing slices of Dask arrays for rejuvenation workers...")
        current_trajectories_np = self.trajectories[:self.M_theta, :, :, :].compute(scheduler='threads') 
        current_ancestors_np = self.ancestors[:self.M_theta, :, :, :].compute(scheduler='threads')     

        self.log("Submitting rejuvenation tasks to joblib.Parallel...")
        with Manager() as manager:
            tracker = manager.list()
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._rejuvenate_one_theta)(
                    i, rejuvenate_A, tracker,
                    current_trajectories_np[i],
                    current_ancestors_np[i]
                )
                for i in range(self.M_theta) 
            )
            self.log(f"Rejuvenation tasks completed by {len(set(tracker))} workers.")

        self.log("Storing rejuvenated particles into Zarr...")

    def _smc_inner_loop(self, i, t, k, Phi_k, A_k, Api_k, y_t, z_t, all_log_lambda_prev, all_ans_prev):
        # single step for the ith theta particle
        if t == 0:
            log_lambda_prev = all_log_lambda_prev[i]
        else:
            idx = all_ans_prev[i]
            log_lambda_prev = all_log_lambda_prev[i,idx]

        log_lambda_t = log_lambda_prev + np.sqrt(Phi_k[i]) * np.random.normal(size=self.N_x)
        log_lambda_t = np.clip(log_lambda_t, -10.0, 10.0)
        #self.trajectories[i, :, k, t + 1] = log_lambda_t

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
            #self.ancestors[i, :, k, t] = new_idx
            self.inner_weights[i, :, k] = np.zeros(self.N_x)
        else:
            #self.ancestors[i, :, k, t] = np.arange(self.N_x)
            new_idx = np.arange(self.N_x)
        
        return ll_i, log_lambda_t, new_idx
    

    def smc_step_single_parallel(self, client: Client, t, k): # Add client as an argument

        Phi_k = np.array(self.Phi[:, k]).reshape(-1, 1)
        A_k = np.array(self.A[:, k, :]).reshape(self.N_theta, 1, self.K)
        Api_k = np.array(self.Api[:, k, :]).reshape(self.N_theta, 1, -1)
        y_t_vec = self.y[t].reshape(-1, 1)   # shape: (K, 1)
        z_t_vec = self.z[t].reshape(-1, 1)   # shape: (D, 1)

        
        # CRITICAL: Compute these Dask array slices to NumPy *once* before the parallel loop.
        all_log_lambda_prev = self.trajectories[:, :, k, t].compute(scheduler='single-threaded')
        if t > 0:
            all_ans_prev = self.ancestors[:, :, k, t-1].compute(scheduler='single-threaded').astype(int)
        else:
            all_ans_prev = None # Not used for t=0
        
        self.log(f"Running SMC inner loop for t={t}, k={k} in parallel...")
        # Note: We are passing the *entire* `all_log_lambda_prev` and `all_ans_prev` arrays
        # to each parallel worker. While not ideal for memory if very large, it matches
        # `_smc_inner_loop`'s signature, and `joblib`'s threading backend
        # might handle shared read-only data efficiently.
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._smc_inner_loop)(
                i, t, k, Phi_k, A_k, Api_k, y_t_vec, z_t_vec, all_log_lambda_prev,
                all_ans_prev
                )
            for i in range(self.N_theta) # Iterate over all theta particles
        )

        # Unpack results. Each result from _smc_inner_loop is (ll_i, log_lambda_t, new_idx)
        ll_k_list, new_trj_list, new_ans_list = zip(*results)

        ll_k = np.array(ll_k_list).reshape(-1, 1) # This is (N_theta, 1)

        # --- MODIFICATION ---
        # REMOVE the da.store() calls and replace them with a direct write loop.
        # This avoids sending the write task to the fragile Dask workers.

        self.log(f"Writing results for t={t}, k={k} directly to Zarr...")
        # We loop through the results and write them one by one.
        # self.store_trajectories is the raw Zarr array object.
        for i in range(self.N_theta):
            # Write the new trajectory for particle i
            self.store_trajectories[i, :, k, t + 1] = new_trj_list[i]
        
            # Write the new ancestors for particle i
            self.store_ancestors[i, :, k, t] = new_ans_list[i]
    
        self.log(f"Direct writes for t={t}, k={k} complete.")

        # Compute marginal likelihood
        outer_norm = self.outer_weights - scipy.special.logsumexp(self.outer_weights)
        marginal_ll = scipy.special.logsumexp(outer_norm + np.squeeze(ll_k))
        if marginal_ll > 0:
            self.log(f"[Warning] marginal_ll = {marginal_ll:.4f} > 0 at t={t}, likely numerical issue. Max ll_k: {np.max(ll_k)}, max outer weight: {np.max(outer_norm)}")
       
        # Update outer weight
        self.outer_weights += np.squeeze(ll_k)
        # Normalize and compute ESS
        self.outer_weights -= np.max(self.outer_weights)

        outer_weight = np.exp(self.outer_weights)
        outer_weight /= np.sum(outer_weight) if np.sum(outer_weight) > 0 else self.N_theta # Handle all zero weights
        ess_k = 1.0 / np.sum(outer_weight**2)

        return marginal_ll, ess_k
    

    def run(self, client, y, t_init=0, init_result=None, if_pred=False):
        """
        Executes the Waste-Free SMC^2 algorithm.

        Args:
            client (dask.distributed.Client): The Dask client object for distributed computation.
            y (array-like): The observed data.
            t_init (int, optional): Initial time step. Defaults to 0.
            init_result (dict, optional): Result from a previous run for initialization. Defaults to None.
            if_pred (bool, optional): Whether to compute predictive residuals. Defaults to False.
        """

        #self.init_prior(y)

        self.ESS_track = []
        self.likelihood_track = []
        self.predicted_residuals = []

        if t_init == 0:
            self.initialize_from_prior(self.y)
        else:
            if init_result is None:
                # Assuming initialize_from_posterior does not require client directly,
                # or that it handles Dask operations internally if needed.
                init_result = initialize_from_posterior(y, self.p, t_init, self.N_rejuvenate, self.N_x, self.M_theta)
            self.intialize_from_posterior_result(init_result, t_init, client)
        

        for t in range(t_init, self.T):
            self.current_t = t
            self.log(f'at time {t}')

            for k in range(self.K):
                self.current_k = k
                # Pass the Dask client to the smc_step_single_parallel method
                marginal_ll, ess_k = self.smc_step_single_parallel(client, t, k) 
                self.ESS_track.append(ess_k)
                
                self.log(f'at time t = {t}, step = {k}, marginal log_likelihood = {marginal_ll}')
                self.likelihood_track.append(marginal_ll)
                self.log(f'ess at step {k} {ess_k}')

                if ess_k < self.threshold * self.N_theta:
                    # Pass the Dask client to the rejuvenate_parallelized method
                    self.rejuvenate_parallelized(client=client) 

            self.log(f'final ess {ess_k}')
            
            if if_pred:
                pred = self.monte_carlo_predictive_residuals()
                self.log(f'at time t = {t}, monte_carlo_predictive_residuals = {pred}')
                self.predicted_residuals.append(pred)

