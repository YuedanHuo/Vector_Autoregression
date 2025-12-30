import numpy as np
from numba import njit
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

import tracemalloc
tracemalloc.start()



sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from prior_setup import BayesianVARPrior


# import the tree object
#sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/other code/path storage')
#import tree as Tree_class

sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/Gibbs_with_numba')
from Phi_update import compute_posterior_phi_by_component_numba
from CSMC import run_csmc_full_numba
from APi_update import update_APi_numba
from A_update_numba import compute_posterior_A_with_log_lambdas_numba

sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/Gibbs_with_numba/pybind11_tree.h')
import tree as Tree_class

sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/SMC_square')
from Initialization import initialize_from_posterior

# Replacement for Scipy LogSumExp with normalization
@njit(nogil=True)
def numbafied_logsumexp_and_normalize(logw):
    """Calculates weights from log-weights safely."""
    # logw shape: (N_x, K)
    N, K = logw.shape
    weights = np.empty_like(logw)
    
    for k in range(K):
        max_val = -np.inf
        for n in range(N):
            if logw[n, k] > max_val:
                max_val = logw[n, k]
        
        sum_exp = 0.0
        for n in range(N):
            val = np.exp(logw[n, k] - max_val)
            weights[n, k] = val
            sum_exp += val
            
        # Normalize
        for n in range(N):
            weights[n, k] /= sum_exp
            
    return weights


# helper function for generating synthetic data in rejuvenation
@njit(nogil=True)
def generate_synthetic_data(current_t, current_k, K, fix_log_lambda, Ay, A, Pi, z, Phi):
    """
    Generates the synthetic data for the A update entirely in Numba.
    """
    syn_lambda = fix_log_lambda.copy()
    syn_Ay = Ay.copy()
    
    # Pre-calculate deterministic mean part
    APiz = A @ Pi @ z[current_t]
    
    # Loop over unobserved components
    for k in range(current_k + 1, K):
        # 1. Simulate Log Lambda (Random Walk)
        prev_lambda = syn_lambda[k, current_t - 1]
        noise_lambda = np.random.normal(0, 1) * np.sqrt(Phi[k])
        syn_lambda[k, current_t] = prev_lambda + noise_lambda
        
        # 2. Simulate Ay
        lambda_val = np.exp(syn_lambda[k, current_t])
        noise_y = np.random.normal(0, 1) * np.sqrt(lambda_val)
        syn_Ay[current_t, k] = APiz[k] + noise_y
        
    return syn_lambda, syn_Ay

import numpy as np
from numba import njit

@njit(nogil=True, cache=True)
def numba_logsumexp(a):
    """
    Numba-compatible logsumexp implementation.
    Releases GIL and avoids SciPy overhead.
    """
    # Find max for numerical stability
    max_val = np.max(a)
    
    if np.isinf(max_val):
        return -np.inf

    # 2. Sum exponentials
    sum_exp = 0.0
    for x in a:
        sum_exp += np.exp(x - max_val)
    
    return np.log(sum_exp) + max_val

@njit(nogil=True, cache=True)
def smc_math_kernel(log_lambda_prev, Phi_val, A_flat, Api_flat, 
                    y_t, z_t, inner_weights, N_x, threshold):
    """
    The heavy lifter. Performs Proposal -> Weighting -> Resampling Decision.
    """
    # propose
    noise = np.random.normal(0.0, 1.0, N_x)
    log_lambda_t = log_lambda_prev + np.sqrt(Phi_val) * noise
    
    # Clip in place
    log_lambda_t = np.clip(log_lambda_t, a_min=-10.0, a_max=10.0)

    # residual calculation
    # Manually compute dot products for speed on small vectors
    # res = (A @ y_t - Api @ z_t)
        
    res = A_flat @ y_t - Api_flat @ z_t
    #res = np.clip(res, -1e3, 1e3) # Clip residual to prevent extreme values.
    if res < -1e3: res = -1e3
    if res > 1e3:  res = 1e3
        

    # --- 3. WEIGHTING ---
    # Vectorized operation in Numba (uses SVML, very fast)
    # lw = -0.5 * (log(2pi) + log_lam + exp(-log_lam)*res^2)
    const_term = np.log(2 * np.pi)
    
    # Calculate new incremental weights
    new_lw = -0.5 * (const_term + log_lambda_t + np.exp(-log_lambda_t) * res ** 2)
    
    new_lw = np.where(np.isnan(new_lw), -np.inf, new_lw)
    inner_weights += new_lw
        
    # --- 4. LOG LIKELIHOOD ---
    # Use our custom logsumexp
    log_sum_w = numba_logsumexp(inner_weights)
    ll_i = log_sum_w - np.log(N_x)

    # --- 5. NORMALIZE & ESS ---
    # Normalize weights in log domain first
    # We update inner_weights in-place for the return
    inner_weights -= log_sum_w
    
    # Exponentiate
    weights = np.exp(inner_weights)
    
    # Safety normalize (linear domain)
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum
    else:
        weights[:] = 1.0 / N_x

    # Calculate ESS
    ess = 1.0 / np.sum(weights**2)
    
    # --- 6. RESAMPLING DECISION ---
    do_resample = False
    new_idx = np.arange(N_x) # Default identity
    
    if ess < threshold * N_x:
        do_resample = True
        
        # Multinomial Resampling using Inverse CDF (Fastest in Numba)
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0 # Fix precision
        u = np.random.rand(N_x)
        # Numba supports searchsorted
        new_idx = np.searchsorted(cdf, u)
        
        # Reset weights to 0 (log scale)
        inner_weights[:] = 0.0
        
        # Resample the log_lambda_t immediately to save Python from doing it
        # Actually, let's just return indices and let Python/C++ handle the update
        # to keep the interface simple. 
        # But we MUST permute log_lambda_t here to return the correct 'current' state
        log_lambda_t = log_lambda_t[new_idx]

    return ll_i, log_lambda_t, new_idx, do_resample

from numba import prange
@njit(parallel=True, fastmath=True)
def predict_numba_kernel(indices, A_all, Pi_all, Phi_all, 
                         start_log_lambdas, initial_z, y_future, 
                         step_ahead, K, p):
    """
    Numba kernel to handle the Monte Carlo simulation loop in parallel.
    """
    num_samples = len(indices)
    
    # Outputs
    # shape: (num_samples, step_ahead, K)
    predicted_next_y = np.zeros((num_samples, step_ahead, K)) 
    # shape: (num_samples, step_ahead)
    log_score_next_y = np.zeros((num_samples, step_ahead)) 
    
    # Loop over samples in parallel
    for i in prange(num_samples):
        # Retrieve parameters for this particle
        # Note: We pass the full arrays and index into them to avoid 
        # creating copies in Python before the call
        idx = indices[i]
        
        # Make local copies 
        Pi_i = Pi_all[idx]
        A_i = A_all[idx]
        Phi_val = Phi_all[idx] # shape (K,)
        
        # Volatility is diagonal, so sqrt(Phi) is just sqrt of vector
        vol_scale = np.sqrt(Phi_val)
        
        # Start State
        curr_log_lambda = start_log_lambdas[i].copy()
        
        # Local copy of Z for recursion
        syn_z = initial_z.copy()
        
        # Step Ahead Loop
        for step in range(step_ahead):
            
            # Volatility Random Walk 
            noise = np.random.randn(K) * vol_scale
            curr_log_lambda += noise
            
            # Point Prediction (Conditional Mean)
            # E[y_{t+1}] = Pi * z_t
            # Pi is (K, 1+pK), syn_z is (1+pK,)
            # Manual dot product or matrix-vector mult
            expected_y = Pi_i.dot(syn_z)
            
            predicted_next_y[i, step, :] = expected_y
            
            # Log Score (Likelihood of FUTURE Real Data)
            # Get real future observation (passed from Python)
            # y_future shape: (step_ahead, K)
            y_real = y_future[step] 
            
            # Residual: A * y_real - A * Pi * z
            # Note: A * Pi * z is just A * expected_y
            term1 = A_i.dot(y_real)
            term2 = A_i.dot(expected_y)
            res = term1 - term2
            
            # Clip for numerical safety
            for k_r in range(K):
                if res[k_r] < -1000.0: res[k_r] = -1000.0
                if res[k_r] > 1000.0:  res[k_r] = 1000.0
            
            # Log-Likelihood Calculation (Vectorized)
            # ll = -0.5 * (log(2pi) + lambda + res^2 * exp(-lambda))
            # Precomputed constant log(2pi) approx 1.837877
            ll_sum = 0.0
            for k_ll in range(K):
                lam = curr_log_lambda[k_ll]
                r = res[k_ll]
                ll_sum += -0.5 * (1.837877 + lam + (r * r) * np.exp(-lam))
                
            log_score_next_y[i, step] = ll_sum
            
            #  Update Z for next step
            if step < step_ahead - 1:
                # shift lag values in syn_z
                prev_lags = syn_z[1:-K] # Drops the oldest lag
                
                # Construct new lag vector
                # [new_y, prev_lags]
                
                # Update syn_z in place
                # intercept stays same at index 0
                
                # Move old data to the right
                syn_z[K+1:] = syn_z[1:-K]
                
                # Insert new prediction at the front (indices 1 to K)
                syn_z[1:K+1] = expected_y

    return predicted_next_y, log_score_next_y

class WF_SMC_Square_with_tree:
    """
    Implements a Waste-Free Sequential Monte Carlo (SMC^2) algorithm for a VAR model.
    This class combines Gibbs sampling for parameters (A, Pi, Phi) with Conditional SMC (CSMC)
    for sampling log-lambda trajectories, as described in the accompanying algorithm document.

    Update for the option to do intermediate rejuvenation for A or not. One can choose different scheme by setting update_A:
    1. if True, then we do intermediate rejuvenation for A.
    2. if False, we update according to two different ESS threshold when observe/ not observe full data,
    and only update A when full data observed. 
    """
    def __init__(self, p, logfile, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.4, threshold_A = 0.6, update_A = True):
        """
        Initializes the WF_SMC_Square sampler with model parameters and SMC settings.
        
        Args:
            p (int): The number of lags for the VAR model.
            logfile (str): Path to the log file for recording progress and warnings.
            N_theta (int): Number of outer particles for the parameters (A, Pi, Phi).
                           These are often referred to as 'theta particles' or 'parameter particles'.
            N_x (int): Number of inner particles for each Conditional SMC (CSMC) chain.
                       These inner particles are used to sample log-lambda trajectories for each theta particle.
            N_rejuvenate (int): Number of rejuvenation steps performed for each resampled theta particle.
            threshold (float): ESS (Effective Sample Size) threshold (as a proportion of N_theta or N_x)
                               that triggers resampling/rejuvenation.
        """
        self.p = p # number of lag for the VAR model 
        self.N_theta = N_theta # number of theta particles (parameter particles)
        self.N_x = N_x # number of inner particles for each theta particle
        self.threshold = threshold # ESS threshold (proportion) to trigger resampling/rejuvenation
        # in the case that we skip intermediate rejuvenation for A
        # use a different threshold after observing full data
        self.threshold_A = threshold_A 
        self.N_rejuvenate = N_rejuvenate # number of rejuvenation steps per resampled particle
        self.logfile = logfile # path to the log file
        
        # M_theta: For waste-free SMC, this defines the number of particles to be resampled
        # such that M_theta * (N_rejuvenation + 1) = N_theta
        self.M_theta = N_theta // (N_rejuvenate + 1)

        self.mem_usage =[] # log to track memory usage during execution

        self.update_A = update_A # indicator for whether we do intermediate update for A

    def log(self, message):
        """
        Appends a message to the specified log file.
        
        Args:
            message (str): The string message to be logged.
        """
        with open(self.logfile, 'a') as f:
            f.write(message + '\n')
            
    def init_prior(self, y):
        '''
        Sets up priors for the VAR model parameters (A, Pi, Lambda, Phi) and initializes
        related data structures based on the model overview and prior specifications
        in the algorithm document. 
        
        Args:
            y (np.ndarray): The observed time series data, typically of shape (T, K),
                            where T is the number of time steps and K is the number of variables.
        '''
        # Set up priors for A, Pi, and Phi based on the Bayesian VAR model.
        # The 'BayesianVARPrior' class handles the prior distributions
        # as described in Section 2 (A), Section 3 (Pi), Section 4 (Lambda), and Section 5 (Phi).
        bayesian_var_prior = BayesianVARPrior(y, p=self.p, sigma_0=2.31, threshold=0.5)
        priors = bayesian_var_prior.get_priors()
        
        # Priors for A (Lower triangular matrix with ones on diagonal) 
        self.Sigma_A = priors["Sigma_A"] # Covariance matrix for the Gaussian prior on each row of A. 
        self.mu_A = priors["mu_A"]       # Mean vector for the Gaussian prior on each row of A.
        
        # Priors for Pi 
        self.Pi_prior_mean = priors["Pi_prior_mean"] # Mean of the Minnesota prior for Pi. [cite: 29]
        self.Pi_prior_var_rows = priors["Pi_prior_var"] # Variance of the Minnesota prior for Pi (row-wise). 
        self.Pi_prior_var_rows_inv = priors["Pi_prior_var_inv"] # Inverse of the Pi prior variance (for computational efficiency).
        
        # Prior for initial log-lambda (sigma_0 corresponds to the variance of log_lambda_0,j) 
        self.sigma_0 = priors["sigma_0"]

        # Processed input data and dimensions
        self.z = priors["Z"] # Transformed input data 'z_t' from the VAR model definition. 
        self.y = priors["y"] # Observed data 'y_t' used in the model. 
        self.T, self.K = self.y.shape # T: number of time steps, K: number of variables in y_t. 
        print(self.T) 
        
        # Initial Phi (diagonal matrix representing variance of log-lambda increments) 
        self.Phi = priors['phi'] 
        
        # Initialize storage for SMC-Square particles and trajectories
        # outer_weights: Weights for each of the N_theta parameter (theta) particles.
        self.outer_weights = np.zeros((self.N_theta,))
        
        # inner_weights: Weights for the inner SMC chains. Each theta particle has K independent CSMC chains,
        # and each chain has N_x particles. Shape (N_theta, N_x, K).
        self.inner_weights = np.zeros((self.N_theta, self.N_x, self.K)) 

        self.trees = [[Tree_class.Tree(self.N_x, 10*self.N_x, 1) for k in range(self.K)] for i in range(self.N_theta)]
        
        # Initialize ancestor indices for the very first step (t=0) to be simply 0 to N_x-1,
        # implying no resampling at initialization.
        #for i in range(self.N_theta):
        #    for k in range(self.K):
        #        self.ancestors[i, :, k, 0] = np.arange(self.N_x)
        
        # Initialize storage for parameter particles (theta particles)
        # A: Lower triangular matrix with ones on the diagonal. Shape (N_theta, K, K).
        self.A = np.zeros((self.N_theta, self.K, self.K))
        # Pi: Regression coefficients matrix. Shape (N_theta, K, K*p + 1). 
        self.Pi = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        # Api: Product of A and Pi, pre-calculated for efficiency. Shape (N_theta, K, K*p + 1).
        self.Api = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        # Phi: Diagonal elements of the Phi matrix (variance of log-lambda increments). Shape (N_theta, K).
        self.Phi = np.zeros((self.N_theta, self.K))


    def intialize_from_posterior_result(self, results, init_t):
        '''
        Initializes the WF_SMC_Square particles (parameters and trajectories)
        from a set of pre-computed posterior samples, typically obtained from
        M_theta independent run of Particle Gibbs conditioned on observed data up to 'init_t'.
        This is used for advanced initialization, potentially after a burn-in phase.
        
        Args:
            results (list): A list of dictionaries, where each dictionary contains
                            posterior samples for A, Pi, Phi, and corresponding
                            log-lambda trajectories and ancestor indices.
                            Expected to be of length `M_theta` for `N_rejuvenation + 1` samples each,
                            representing independent runs.
            init_t (int): The time step up to which the 'results' are conditioned on observed data y.
                          The trajectories and ancestors will be initialized up to this time point.
        '''
        # Iterate through each independent run result (M_theta runs)
        for i in range(self.M_theta):
            sample_result = results[i]
            # Distribute the N_rejuvenation + 1 samples from each run to the target_index slots
            for j in range(self.N_rejuvenate + 1):
                target_index = i * (self.N_rejuvenate + 1) + j
                # Assign sampled A, Pi, Phi parameters to the corresponding theta particles
                self.A[target_index] = sample_result['A'][j]
                self.Pi[target_index] = sample_result['Pi'][j]
                self.Phi[target_index] = sample_result['Phi'][j]
                # Recompute Api for the newly assigned A and Pi
                self.Api[target_index] = self.A[target_index] @ self.Pi[target_index]
            
                # Loop through T to create tree object
                # Load log-lambda trajectories into Trees
                for k in range(self.K):
                    tree_obj = self.trees[target_index][k]

                    # Extract the trajectory up to init_t + 1
                    # skip the initialized value
                    traj_k = sample_result['trajectories'][j, :, k, 1 :init_t + 1]  # shape (N_x, init_t+1)
                    
                    # Initialize Tree with the first step
                    tree_obj.init(traj_k[:, [0]].T)  # shape (1, N_x)

                    # Insert the remaining steps
                    for t in range(1, init_t-1): # start from 1 as we have init
                        x_t = traj_k[:, [t]].T  # shape (1, N_x)
                        ancestors_t = sample_result['ancestors'][j, :, k, t].astype(int)
                        tree_obj.insert(x_t, ancestors_t)
                    # for the last time step
                    # call update to prune the tree
                    x_t = traj_k[:, [init_t-1]].T 
                    ancestors_t = sample_result['ancestors'][j, :, k, init_t-1].astype(int)
                    tree_obj.update(x_t, ancestors_t)


    def waste_free_resampling(self):
        '''
        Performs the waste-free resampling step for the outer (theta) particles.
        This process selects M_theta particles based on their current outer weights.
        After resampling, the outer weights are reset.
        '''
        # Normalize the outer weights to create a valid probability distribution for resampling.
        # This uses the log-sum-exp trick for numerical stability to prevent underflow/overflow.
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)
        
        # Sample M_theta indices from the current N_theta particles based on their probabilities.
        # np.searchsorted with np.cumsum provides an efficient way to perform multinomial sampling.
        indices = np.searchsorted(np.cumsum(probs), np.random.rand(self.M_theta))
        
        # Reassign the first M_theta slots of all particle-related storage
        # (A, Pi, Api, Phi, trajectories, ancestors, inner_weights)
        # to the resampled particles. The remaining N_theta - M_theta slots
        # will be filled by rejuvenated particles.
        self.A[:self.M_theta] = self.A[indices]
        self.Pi[:self.M_theta] = self.Pi[indices]
        self.Api[:self.M_theta] = self.Api[indices]
        self.Phi[:self.M_theta] = self.Phi[indices]

        # Resample Trees
        new_trees = []
        for i in range(self.M_theta):
            theta_index = indices[i]
            # Copy the list of K trees for this particle
            new_trees.append([self.trees[theta_index][k].copy() for k in range(self.K)])
        self.trees[:self.M_theta] = new_trees
        
        #######################DELETE FOR TESTING
        # empty the list so that the old trees are ready for garbage collection
        #for i in range(self.M_theta, self.N_theta):
        #    self.trees[i] = [[] for _ in range(self.K)]
        
        # Outer weights are reset to zero (on log scale) after resampling,
        # as these resampled particles are now considered to have equal initial weight
        # for the next cycle of importance sampling.
        self.outer_weights = np.zeros(self.N_theta)
        
        # Inner weights for the resampled particles are also reassigned,
        # maintaining the relative weights within each inner CSMC chain.
        self.inner_weights[:self.M_theta] = self.inner_weights[indices]

    
    def _rejuvenate_one_theta(self, i, rejuvenate_A, tracker):
        '''
        Performs a single rejuvenation step for the i-th theta (parameter) particle.
        This function implements the Particle Gibbs sampling steps for A, Pi, Phi, and log-lambda
        trajectories for one particle.
        At this point, y_{0:current_t-1,0:K-1} and y_{current_t, 0:current_k} are observed.
        
        Args:
            i (int): Index of the theta particle to be rejuvenated (from 0 to M_theta - 1).
            tracker (multiprocessing.manager.list or None): A list to track worker IDs for parallelization,
                                                              used for debugging/monitoring parallel execution.
        '''

        # Add tracking of resources for parallel execution (if tracker is provided)
        if tracker is not None:
            tracker.append(threading.get_ident())  
        
        # Take the log inner weights of the i-th theta particle for its K independent SMC chains.
        # Shape: (N_x, K)
        logw = self.inner_weights[i]
        
        # Normalize each of the K SMC chains' inner weights (log scale)
        # using the log-sum-exp trick for numerical stability.
        weights = numbafied_logsumexp_and_normalize(logw)
        
        # Initialize storage for the K sampled log-lambda trajectories.
        # These fixed trajectories are used for rejuvenating Pi parameters.
        # 'np.inf' is used as a placeholder for unobserved/future log-lambdas ang y to effectively
        # zero out their contribution in certain calculations.
        fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf)
        
        # Helper array to track valid lengths for Phi update
        # Instead of a list of arrays, we pass the full matrix and this length vector
        valid_lengths_phi = np.zeros(self.K, dtype=np.int32)   
        
        # Sample backward for the K initial fixed trajectories (one for each k component).
        # This fixes a path of log-lambdas for use in the subsequent Gibbs updates for A, Pi, Phi.
        for k in range(self.K): # Iterate over each of the K variables/components.
            # Sample an index for the fixed trajectory from the inner particles based on their weights.
            idx = np.random.choice(self.N_x, p=weights[:, k])
            tree = self.trees[i][k]  # access the Tree object for theta i, component k
            if k <= self.current_k:
                # For components 'k' that have already been observed up to 'current_t':
                # The degree of freedom for Phi_k posterior is (current_t + 1) because
                # log-lambdas are observed from t=0 to t=current_t.
                valid_lengths_phi[k] = self.current_t + 1
                #use the sample path function to get the path for the idx inner particle
                # the length of the path should coincide with the length of observed data
                fix_log_lambda_i[k, :self.current_t + 1] = tree.get_path(idx)
            else:
                # For components 'k' that are unobserved at 'current_t' (i.e., k > current_k):
                # We only have observations up to 'current_t - 1'.
                # The degree of freedom for Phi_k posterior is 'current_t'.
                valid_lengths_phi[k] = self.current_t 
                fix_log_lambda_i[k, :self.current_t] = tree.get_path(idx)

        # Initialize the parameters for this specific theta particle from its current state.
        A_i = self.A[i].copy()
        Pi_i = self.Pi[i].copy()
        Phi_i = self.Phi[i].copy()
        Ay = self.y @ A_i.T 

        # Perform N_rejuvenation steps of Gibbs sampling.
        for j in range(self.N_rejuvenate):
            # Determine the target index where the rejuvenated particle's parameters will be stored.
            # This is part of the waste-free scheme, where rejuvenated particles fill new slots.
            target_idx = self.M_theta + i * self.N_rejuvenate + j

            # Rejuvenation of Pi 
            # Updates Pi_i conditioned on A_i, fixed log-lambda paths, y, and z.
            # Refer to Section 3 "Pi", specifically Eq. (59) and (60) for posterior mean and covariance.
            # Note: The `update_APi` function is assumed to implement this update logic.
            # We pass in y and z till time current_t
            # for the k such that y_{current_t,k} unobserved, fix_log_lambda_i at the correspond position
            # is np.inf, which will cancel out and contribute 0
            Pi_i = update_APi_numba(
                Ay[:self.current_t + 1], # Observed Ay up to current_t.
                self.z[:self.current_t + 1], # Observed z up to current_t.
                Pi_i, # Current Pi_i for potential conditioning.
                fix_log_lambda_i, # Fixed log-lambda trajectories.
                self.p, # Lag order.
                A_i, # Current A_i.
                self.Pi_prior_mean, # Global Pi prior mean.
                self.Pi_prior_var_rows_inv # Global Pi prior inverse variance.
            )
            
            # Rejuvenation step for Phi (variance of log-lambda increments)
            # Updates Phi_i conditioned on the sampled log-lambda trajectories.
            # Refer to Section 5 "Phi", specifically Eq. (101) for posterior Inverse Gamma distribution.
            Phi_i = compute_posterior_phi_by_component_numba(
                fix_log_lambda_i, # Pass the full 2D array
                valid_lengths_phi, # Pass the lengths (int array)
                K=self.K
            )
            # Update Api (A @ Pi) with the newly sampled Pi_i.
            Api_i = A_i @ Pi_i

            # Rejuvenation of A (Lower triangular matrix)
            # we rejuvenate A everytime if we want to do the intermediate step, indicated by update_A
            if self.update_A == True: 
                # For unobserved y_{current_t, k} (where k > current_k), we need to simulate them
                # to complete the y matrix for the A update.
                # simulate by a outer helper function that is jit
                syn_fix_log_lambda_i, syn_Ay = generate_synthetic_data(
                    self.current_t, self.current_k, self.K,
                    fix_log_lambda_i, Ay, A_i, Pi_i, self.z, Phi_i
                )
            
                # Construct a 'synthetic_y' matrix for the A update,
                # which combines observed y (up to current_t, current_k) with simulated y (at current_t, for k > current_k).
                # Note: A_i.T is (K,K), np.linalg.inv(A_i.T) is (K,K). This reconstructs y from Ay.
                syn_y = syn_Ay[:self.current_t + 1] @ np.linalg.inv(A_i.T)
            
                # Now, update A_i using the synthetic_y matrix.
                # Note: The `compute_posterior_A_with_log_lambdas` function is assumed to implement this.
                A_i = compute_posterior_A_with_log_lambdas_numba(
                    syn_y, # The (partially simulated) y matrix.
                    self.z[:self.current_t + 1], # z up to current_t.
                    syn_fix_log_lambda_i, # Fixed log-lambda trajectories.
                    self.mu_A, self.Sigma_A, # Priors for A.
                    Pi_i # Current Pi_i.
                )
            # if intermediate step skipped, then we only do it when full data observed
            # indicated by rejuvenate_A
            elif rejuvenate_A == True: 
                A_i = compute_posterior_A_with_log_lambdas_numba(
                    self.y[:self.current_t + 1],
                    self.z[:self.current_t + 1],
                    fix_log_lambda_i,
                    self.mu_A,
                    self.Sigma_A,
                    Pi_i
                )
            # After A is updated, recompute Ay using the real observed data (self.y)
            # and the newly sampled A_i. This discards any simulated y values for consistency.
            Ay = self.y @ A_i.T
            

            # Rejuvenation of log-lambda trajectories using Conditional Sequential Monte Carlo (CSMC)
            # This loops through each of the K components, running a CSMC for each.
            for k in range(self.K):
                # Determine the observation time window for CSMC based on whether 'k' has been observed at current_t.
                t_k = self.current_t + 1 if k <= self.current_k else self.current_t
                
                # replace the CSMC class by pure numba function
                # Note: Pass only the specific slices needed (e.g. Phi_i[k] instead of Phi_i)
                trajectory, ancestor, sampled_traj, _, _ = run_csmc_full_numba(
                    Num=self.N_x,
                    phi_val=Phi_i[k],         # Pass scalar
                    sigma0=self.sigma_0,
                    y_col=Ay[:t_k, k],        # Pass specific column as 1D array
                    z=self.z[:t_k],
                    B_row=Api_i[k, :],        # Pass specific row as 1D array
                    fixed_particles=fix_log_lambda_i[k, :t_k],
                    ESSmin=0.5
                )

                # access the corresponding tree
                ##################################################################
                # recreate an object for now. fail to make the reset function work
                #self.trees[target_idx][k] = Tree_class.Tree(self.N_x, 10*self.N_x, 1) # clear the previous particle clouds
                #self.trees[target_idx][k].reset()
                #self.trees[target_idx][k].init(trajectory[:, [0]].T)  # 1 x N_x
   
                # Insert the remaining steps into trees
                #for t in range(1, len(sampled_traj)-1): # start from 1 as we have init
                #    x_t = trajectory[:, [t]].T  # shape (1, N_x)
                #    ancestors_t = ancestor[:,t].T.astype(int)
                #    self.trees[target_idx][k].insert(x_t, ancestors_t)
                # for the last time step
                # call update to prune the tree
                #x_t = trajectory[:, [len(sampled_traj)-1]].T  # shape (1, N_x)
                #ancestors_t = ancestor[:,len(sampled_traj)-1].T.astype(int)
                #self.trees[target_idx][k].update(x_t, ancestors_t)
                
                fix_log_lambda_i[k, :t_k] = sampled_traj # Update the fixed path used for A/Pi rejuvenation.

                # 2. Prepare the FULL matrices (Force contiguous)
                # the returned shape from CSMC is (N_x, T)
                # C++ is expecting (1, N_x * T) for trajectory and (N_x, T) for ancestor
                full_traj_c = np.ascontiguousarray(trajectory.T.reshape(1,-1), dtype=np.float64)
                full_anc_c = np.ascontiguousarray(ancestor)

                self.trees[target_idx][k].bulk_load(full_traj_c, full_anc_c)

            
            # Store the rejuvenated parameters at their designated 'target_idx' slot.
            self.A[target_idx] = A_i
            self.Pi[target_idx] = Pi_i
            self.Phi[target_idx] = Phi_i
            self.Api[target_idx] = Api_i
            
            # Reset inner weights for the rejuvenated particle to zero (on log scale).
            # This is because the CSMC algorithm performs resampling at its end,
            # resulting in N_x particles with (effectively) equal weights within each chain.
            self.inner_weights[target_idx] = np.zeros((self.N_x, self.K))

    def rejuvenate_parallelized(self, rejuvenate_A=False):
        """
        Orchestrates the parallelized rejuvenation process for the theta particles.
        This function first performs waste-free resampling, then parallelizes the
        '_rejuvenate_one_theta' calls across multiple CPU cores/threads, and finally
        applies a safe replacement mechanism to handle any invalid particles.
        """

        # Perform waste-free resampling of theta particles.
        # This step selects M_theta particles to be rejuvenated based on their outer weights,
        # ensuring computational focus on more promising parameter sets.
        self.waste_free_resampling()
        
        tracker = []
            
        # Execute '_rejuvenate_one_theta' in parallel for each of the M_theta particles.
        # backend="threading" is chosen, indicating thread-based parallelism.
        Parallel(n_jobs=-1, backend="threading")(
                delayed(self._rejuvenate_one_theta)(i, rejuvenate_A, tracker) # Call _rejuvenate_one_theta for each index 'i'
                for i in range(self.M_theta) # Iterate through the M_theta particles to be rejuvenated
        )

        # Log the number of unique parallel workers used during rejuvenation.
        unique_workers = set(tracker)
        self.log(f"[✓] Rejuvenation complete. Used {len(unique_workers)} parallel workers.")

   
    def run(self, y, t_init=0, init_result=None, if_pred=False, step_ahead = 1):
        """
        Executes the main loop of the WF_SMC_Square algorithm.
        This method orchestrates the entire SMC^2 process, including
        initialization, the main time loop, performing SMC steps for
        each variable at each time point, and triggering rejuvenation.
        
        Args:
            y (np.ndarray): The observed time series data, shape (T, K).
            t_init (int): The initial time step to start the main loop from.
                          If 0, initialization is done from prior.
                          If > 0, initialization is done from a posterior result conditioned on y[0:t_init].
            init_result (dict, optional): Dictionary containing posterior samples
                                         for A, Pi, Phi, trajectories, and ancestors
                                         to initialize from a previous run (if t_init > 0).
                                         Defaults to None, in which case `initialize_from_posterior`
                                         is called to generate it.
            if_pred (bool): If True, computes Monte Carlo predictive residuals at each time step.
            step_ahead : prediction step_ahead, expect int >0 & < self.p
        """
        
        # Set up prior values and initialize parameters for the model.
        # This populates self.A, self.Pi, self.Phi, self.trajectories, etc., based on priors.
        self.init_prior(y)

        # Initialize tracking lists for algorithm diagnostics.
        self.ESS_track = []         # Tracks Effective Sample Size at each step.
        self.likelihood_track = []  # Tracks marginal log-likelihood at each step.
        #self.predicted_residuals = [] # Stores predicted residuals if if_pred is True.
        self.predictions = [] # Stores prediction for y_{t+1} if if_pred is True.
        self.log_score = [] # Store the log score if if_pred is True
        
        # Initialization logic: from prior or from a pre-computed posterior result.
        if t_init == 0:
            # If starting from time 0, initialize all parameters and log-lambda trajectories from their priors.
            self.initialize_from_prior(self.y)
        else:
            # If starting from a later time point (t_init > 0),
            # initialize from a posterior result conditioned on y[0:t_init].
            if init_result is None:
                # If no pre-computed 'init_result' is provided, generate it by calling
                # an external helper function 'initialize_from_posterior' to run the particle gibbs.
                init_result = initialize_from_posterior(y, self.p, t_init, self.N_rejuvenate, self.N_x, self.M_theta)
            # Use the provided/generated 'init_result' to populate the class's particle states.
            self.intialize_from_posterior_result(init_result, t_init)
        
        # Main time loop: Observe y_{t,k} one variable (k) at a time, for each time step (t).
        # This iterates from 't_init' up to 'T-1' (total time steps).
        for t in range(t_init, self.T):
            self.current_t = t # Store the current time step for use in helper functions.
            self.log(f'at time {t}') # Log current time step.

            # Inner loop: Iterate through each of the K variables/components for the current time step 't'.
            for k in range(self.K):
                self.current_k = k # Store the current variable index for use in helper functions.
                
                # Perform a single SMC step for the current (t, k) observation.
                # This updates inner particles (log-lambda trajectories) and their weights,
                # and computes the marginal log-likelihood and ESS for y_{t,k}.
                marginal_ll, ess_k = self.smc_step_single_parallel(t, k)
                
                # Log and track diagnostics.
                self.ESS_track.append(ess_k)
                self.log(f'at time t = {t}, step = {k}, marginal log_likelihood = {marginal_ll}')
                self.likelihood_track.append(marginal_ll)
                self.log(f'ess at step {k} {ess_k}')
                
                # Check ESS and trigger rejuvenation if below threshold.
                if self.update_A == True: # if we intermediate rejuvenation for A
                    if ess_k < self.threshold * self.N_theta:
                       self.rejuvenate_parallelized()
                else: # if we skip intermediate rejuvenation for A
                    # we set a larger threshold when observing full data
                    if k == self.K - 1 and ess_k < self.threshold_A * self.N_theta:
                        self.rejuvenate_parallelized(rejuvenate_A= True)
                    # we set a smaller threshold when not observing full data
                    elif ess_k < self.threshold * self.N_theta:
                        self.rejuvenate_parallelized()
                
                # Track memory usage at each (t,k) step.
                current, peak = tracemalloc.get_traced_memory()
                self.mem_usage.append(current / 1e6) # Log in MB.

            self.log(f'final ess {ess_k}') # Log final ESS for the time step 't'.
            
            # Optional: Perform prediction for future residuals.
            if if_pred and t < self.T - step_ahead: # only do prediction for the time point that we have data 
                pred, log_score = self.monte_carlo_prediction(step_ahead=step_ahead)
                self.predictions.append(pred)
                self.log_score.append(log_score)


    def monte_carlo_prediction(self, num_samples=2000, step_ahead=1):
        # Prepare Weights & Indices
        # Safe LogSumExp for weights
        max_w = np.max(self.outer_weights)
        probs = np.exp(self.outer_weights - max_w)
        probs /= np.sum(probs)
        
        indices = np.random.choice(self.N_theta, size=num_samples, p=probs)
        
        # 2. Pre-Sample Initial Volatilities (Python side)
        # This is the only part we keep in Python because accessing 'trees' objects 
        # is hard in Numba. This runs 2000 times but is just a list comprehension, usually fast enough.
        
        # Optimization: Pre-calculate the inner probabilities matrix (N_theta, N_x)
        # to avoid re-calculating logsumexp 2000 times
        
        start_log_lambdas = np.empty((num_samples, self.K))
        
        # Vectorized extraction is hard due to the tree structure, so we loop efficiently
        # We can optimize the inner weight calculation though
        
        for i, idx in enumerate(indices):
            # Normalize inner weights for this particle 'idx'
            lw = self.inner_weights[idx]
            max_lw = np.max(lw, axis=0) # (K,)
            w = np.exp(lw - max_lw)     # (N_x, K)
            w /= np.sum(w, axis=0)      # Normalize columns
            
            # Sample particle indices for each dimension k
            # (Loop over K is small, e.g. 15)
            for k in range(self.K):
                # Sample one index based on weights w[:, k]
                # retrieving the value from the tree
                chosen_particle_idx = np.random.choice(self.N_x, p=w[:, k])
                val = self.trees[idx][k].retrieve_xgeneration(0)[chosen_particle_idx]
                start_log_lambdas[i, k] = val
        
        # Prepare Arrays for Numba
        # Ensure contiguous arrays for speed
        A_all = np.ascontiguousarray(self.A)
        Pi_all = np.ascontiguousarray(self.Pi)
        Phi_all = np.ascontiguousarray(self.Phi)
        
        # Get the future data we want to score against
        # y_{t+1:t+step_ahead}
        # Slice safely
        t_start = self.current_t + 1
        t_end = t_start + step_ahead
        y_future = self.y[t_start:t_end]
        
        if len(y_future) < step_ahead:
            raise ValueError("Not enough future data for prediction steps.")

        # 4. Call Numba Kernel
        pred_y, log_scores = predict_numba_kernel(
            indices, A_all, Pi_all, Phi_all, 
            start_log_lambdas, self.z[t_start], y_future, 
            step_ahead, self.K, self.p
        )
        
        # 5. Final Aggregation
        mean_pred = np.mean(pred_y, axis=0)
        
        # LogSumExp for the score (Monte Carlo integration in log domain)
        # -(log(sum(exp(scores))) - log(N))
        mean_log_score = -(scipy.special.logsumexp(log_scores, axis=0) - np.log(num_samples))
        
        return mean_pred, mean_log_score
    
    def monte_carlo_prediction_old(self, num_samples=2000, step_ahead = 1):
        """
        Computes Monte Carlo prediction for y_{t+1:t+ step_ahead} based on the current set of
        outer (theta) particles and their corresponding log-lambda trajectories;
        And the log score (negative log likelihood) evaluate at real y_{t+1:t+ step_ahead}
        given the current observation y_{1:t}.
        
        Args:
            num_samples (int): The number of samples to draw for Monte Carlo prediction.
            step_ahead (int) : The number of step ahead for prediction.
        
        Returns:
            point estiamation of E[y_{t+1:t+ step_ahead}] of shape (step_ahead, K)
            density estimation of p(y_t+h | y_{1:t}) for h = 1,..., step_ahead of shape (step_ahead)
        """
        # Normalize outer weights to get probabilities for sampling theta particles.
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)

        # Sample 'num_samples' indices of theta particles based on their probabilities.
        # These selected particles will be used to generate predictive residuals.
        indices = np.random.choice(self.N_theta, size=num_samples, p=probs)
        predicted_next_y = np.empty((num_samples, step_ahead, self.K))
        log_score_next_y = np.empty((num_samples, step_ahead))

        # Generate the predictive y_{t+1:t+ step_ahead} for each sampled theta particle.
        for i in range(num_samples):

            idx = indices[i] # Get the index of the sampled theta particle.
            Pi_i = self.Pi[idx]
            A_i = self.A[idx] # Retrieve the A matrix for this particle.
            Phi_i = self.Phi[idx]
            
            # calculate the inner weight
            logw = self.inner_weights[idx]
            logw -= scipy.special.logsumexp(logw, axis=0, keepdims=True)
            weights = np.exp(logw) # Convert log weights to probabilities for sampling.
            # sample the fixed lambda at current_t according to the inner weight
            # and then sample the lambda for next time t
            log_lambda_next_t = [self.trees[idx][k].retrieve_xgeneration(0).flatten()[np.random.choice(self.N_x, p=weights[:, k])] for k in range(self.K)]+\
                  np.random.multivariate_normal(mean=np.zeros(self.K), cov = np.diag(Phi_i))

            # initialize the lag data used for predicting y
            # updated by the predicted value 
            syn_z = self.z[self.current_t + 1].copy() 

            for step in range(step_ahead):
                # predict y at the next time step
                # E(y_{t+1} \mid y_{t}, \theta), the point estimation
                expected_y_t_plus_1 = Pi_i @ syn_z
                predicted_next_y[i,step,:] = expected_y_t_plus_1

                # predict the log sore evaluated at the real y at the next time step
                ################################
                # but there should not be any unobserved value in the model, so we used the syn_z with predicted value
                # the residual at self.current_t + step + 1
                res = A_i @ self.y[self.current_t + step + 1] - A_i @ Pi_i @ syn_z
                res = np.clip(res, -1e3, 1e3) # Clip residual to prevent extreme values.
                # Compute the log likelihood for each inner particle.
                lw = -0.5 * (np.log(2 * np.pi) + log_lambda_next_t + np.exp(-log_lambda_next_t) * res**2)
                log_score_next_y[i, step] = np.sum(lw)

                if step < step_ahead - 1: # if not the last step, create z matrix for the next prediction
                    lag_block = syn_z[1:].reshape(self.p, self.K)  # shape (p, K) with row0 = newest lag y_t
                    # replace the unobserved data by the point estimation we have 
                    # shift down: oldest disappears
                    lag_block[1:] = lag_block[:-1]
                    lag_block[0] = expected_y_t_plus_1  # newest lag
                    syn_z[1:] = lag_block.reshape(-1)
                
                    # update the log_volatility for the next t
                    log_lambda_next_t +=  np.random.multivariate_normal(mean= np.zeros(self.K), cov = np.diag(Phi_i))

        
        # calculate the mean of all predictions
        mean_pred = np.mean(predicted_next_y, axis=0)
        # take the monte carlo average before taking log
        mean_log_score = -(scipy.special.logsumexp(log_score_next_y, axis=0) - np.log(num_samples))

        return mean_pred, mean_log_score
    
    def _calculate_one_particle_pred(self, i):
        '''
        Calculates the one-step-ahead prediction statistics (Observed Error u_{t+1} and 
        Predicted Variance Sigma_{t+1}) conditional on the parameter particle theta_i.

        Returns:
            tuple: (observed_error_u_{t+1}, predicted_covariance_Sigma_{t+1})
        '''
        # Retrieve parameters
        A_i = self.A[i]
        Phi_i = self.Phi[i]
        A_i_inv = np.linalg.inv(A_i)
        Pi_i = self.Pi[i]
    
        expected_lambda_t_plus_1 = np.zeros(self.K)

        for k in range(self.K):
            #Calculate Posterior Mean log(lambda_t) using inner weights
            logw_k = self.inner_weights[i, :, k]
            weights_k = np.exp(logw_k - scipy.special.logsumexp(logw_k))
            # get the log lambdas particles at the current time t(index as current_t + 1)
            log_lambda_particles_t = self.trees[i][k].retrieve_xgeneration(0).flatten()
        
            hat_log_lambda_t_k = np.sum(weights_k * log_lambda_particles_t)
        
            # Calculate E[lambda_t+1, k] using the R.W. property
            #E[exp(X)]=exp(E[X]+ 1/2Var(X)).
            expected_lambda_t_plus_1[k] = np.exp(hat_log_lambda_t_k + .5 * Phi_i[k])

        # Form Predicted Covariance (Predicted Variance)
        # ie VAR(y_{t+1} \mid y_{t})
        Lambda_exp = np.diag(expected_lambda_t_plus_1)
        Sigma_t_plus_1 = A_i_inv @ Lambda_exp @ A_i_inv.T
        
        # E(y_{t+1} \mid y_{t})
        expected_y_t_plus_1 = Pi_i @ self.z[self.current_t + 1] 

        # Residual 
        prediction_error = self.y[self.current_t + 1] - expected_y_t_plus_1
        # Calculate Standard Deviation of y_{t+1} (sqrt of diagonal of Sigma_{t+1})
        std_dev_y_t_plus_1 = np.sqrt(np.diag(Sigma_t_plus_1))
        # Safeguard against division by zero/near-zero values
        std_dev_y_t_plus_1 = np.where(std_dev_y_t_plus_1 > 1e-10, std_dev_y_t_plus_1, 1e-10)
        # Standardized Residual (Z-score vector)
        # Z_k = (y_k - E[y_k]) / STD(y_k)
        standardized_residual = prediction_error / std_dev_y_t_plus_1

        # return the expected mean, expected variance and residual
        return expected_y_t_plus_1 , Sigma_t_plus_1, standardized_residual
    
    # parallization of the previous functions
    def calculate_conditional_prediction_stats(self):
        """
        Orchestrates the parallel calculation of one-step-ahead prediction statistics
        (Observed Error u_{t+1} and Predicted Variance Sigma_{t+1}) for all N_theta particles.

        This function collects the (u_{t+1}, Sigma_{t+1}) pair for each particle.
        """
        
        # We can only calculate u_{t+1} if y_{t+1} is observed (i.e., t+1 < T)
        if self.current_t + 1 >= self.T:
            self.log("[Warning] Cannot calculate prediction statistics: t+1 is out of sample range.")
            return []

        self.log(f"[✓] Calculating conditional prediction statistics for t={self.current_t}...")

        # Execute '_calculate_one_particle_pred' in parallel for all N_theta particles.
        prediction_results = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._calculate_one_particle_pred)(i)
            for i in range(self.N_theta)
        )

        
        self.log(f"[✓] Prediction statistics calculated successfully.")
        
        # prediction_results is a list of tuples: [(u_1, Sigma_1), (u_2, Sigma_2), ...]
        return prediction_results
    
    
    def _smc_inner_loop(self, i, t, k, Phi_k, A_k, Api_k, y_t, z_t):
        # 1. Retrieve Data from C++ 
        # Returns flat array (N_x,)
        log_lambda_prev = self.trees[i][k].retrieve_xgeneration(0).flatten() 

        # 2. Call Numba Kernel (Releases GIL)
        # We pass slices to ensure dimensionality is flat for the kernel
        # inner_weights is modified in-place!
        ll_i, log_lambda_t, new_idx, do_resample = smc_math_kernel(
            log_lambda_prev, 
            Phi_k[i][0],        # Scalar
            A_k[i].flatten(),   # Flat vector
            Api_k[i].flatten(), # Flat vector
            y_t.flatten(),      # Flat vector
            z_t.flatten(),      # Flat vector
            self.inner_weights[i, :, k], # The array to update
            self.N_x, 
            self.threshold
        )

        # 3. Update C++ Tree 
        # We need to reshape for the C++ signature: (1, N_x) or (dimx, N_x)
    
        #  Ensure contiguous arrays for C++
        vals_c = np.ascontiguousarray(log_lambda_t.reshape(1, -1), dtype=np.float64)
        idx_c = np.ascontiguousarray(new_idx, dtype=np.int32)
    
        self.trees[i][k].update(vals_c, idx_c)
    
        return ll_i
    
    def smc_step_single_parallel(self, t, k):
        """
        Performs a single parallelized SMC step for all N_theta theta particles
        at a given time step 't' and variable component 'k'.
        It calls '_smc_inner_loop' in parallel for each theta particle,
        then aggregates the results to compute the marginal likelihood and update outer weights.
        This function orchestrates the inner CSMC runs for all theta particles for a specific (t,k) observation.
        
        Args:
            t (int): Current time step (from 0 to T-1).
            k (int): Current variable component index (from 0 to K-1).
            
        Returns:
            tuple: A tuple containing:
                - marginal_ll (float): The estimated marginal log-likelihood for y_{t,k}.
                - ess_k (float): The Effective Sample Size (ESS) for the outer (theta) particles.
        """
        # Prepare parameters for parallel processing.
        # Reshape Phi_k, A_k, Api_k to be broadcastable to each parallel worker.
        Phi_k = np.array(self.Phi[:, k]).reshape(-1, 1) # Shape: (N_theta, 1)
        A_k = np.array(self.A[:, k, :]).reshape(self.N_theta, 1, self.K) # Shape: (N_theta, 1, K)
        Api_k = np.array(self.Api[:, k, :]).reshape(self.N_theta, 1, -1) # Shape: (N_theta, 1, Kp+1)
        
        # Reshape y_t and z_t for consistency in _smc_inner_loop arguments.
        y_t_vec = self.y[t].reshape(-1, 1)   # shape: (K, 1)
        z_t_vec = self.z[t].reshape(-1, 1)   # shape: (D, 1)

        # Execute '_smc_inner_loop' in parallel for each of the N_theta particles.
        # This computes the log-likelihood contribution (ll_i) for each theta particle.
        ll_k = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._smc_inner_loop)( # Call _smc_inner_loop
                i, t, k, Phi_k, A_k, Api_k, y_t_vec, z_t_vec # Pass arguments specific to each particle and (t,k)
                )
            for i in range(self.N_theta) # Iterate through all N_theta particles
            )

        ll_k = np.array(ll_k).reshape(-1, 1) # Convert results to numpy array (shape: (N_theta, 1)).

        # Compute the marginal log-likelihood for the observation y_{t,k}.
        # This is done by summing the likelihood contributions of all theta particles,
        # weighted by their current outer weights (using log-sum-exp for stability).
        # outer_norm: Normalized outer weights (log scale) to avoid numerical issues.
        outer_norm = self.outer_weights - scipy.special.logsumexp(self.outer_weights)
        # marginal_ll: Sum of (normalized_outer_weight + inner_ll_k) on log scale.
        marginal_ll = scipy.special.logsumexp(outer_norm + np.squeeze(ll_k))
        
        # Warning for unlikely positive marginal likelihood, indicating potential numerical issues.
        if marginal_ll > 0:
            self.log(f"[Warning] marginal_ll = {marginal_ll:.4f} > 0 at t={t}, likely numerical issue. Max ll_k: {np.max(ll_k)}, max outer weight: {np.max(outer_norm)}")

        # Update outer weights for the next time step.
        # The outer weights are updated by adding the log-likelihood contributions from this step.
        self.outer_weights += np.squeeze(ll_k)
        
        # Normalize outer weights to prevent numerical overflow/underflow and compute ESS.
        self.outer_weights -= np.max(self.outer_weights) # Subtract max for stability.
        outer_weight = np.exp(self.outer_weights) # Convert to linear scale.
        outer_weight /= np.sum(outer_weight) # Normalize to sum to 1.
        
        # Compute Effective Sample Size (ESS) for the outer (theta) particles.
        # ESS = 1 / sum(normalized_outer_weights^2).
        # A low ESS indicates particle degeneracy, requiring rejuvenation.
        ess_k = 1.0 / np.sum(outer_weight**2)

        return marginal_ll, ess_k
