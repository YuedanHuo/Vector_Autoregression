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

import tracemalloc
tracemalloc.start()

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
    """
    Implements a Waste-Free Sequential Monte Carlo (SMC^2) algorithm for a VAR model.
    This class combines Gibbs sampling for parameters (A, Pi, Phi) with Conditional SMC (CSMC)
    for sampling log-lambda trajectories, as described in the accompanying algorithm document.
    """
    def __init__(self, p, logfile, N_theta=50, N_x=300, N_rejuvenate=5, threshold=0.4):
        """
        Initializes the WF_SMC_Square sampler with model parameters and SMC settings.
        
        Args:
            [cite_start]p (int): The number of lags for the VAR model[cite: 8].
            logfile (str): Path to the log file for recording progress and warnings.
            N_theta (int): Number of outer particles for the parameters (A, Pi, Phi).
                           These are often referred to as 'theta particles' or 'parameter particles'.
            N_x (int): Number of inner particles for each Conditional SMC (CSMC) chain.
                       These inner particles are used to sample log-lambda trajectories for each theta particle.
            N_rejuvenate (int): Number of rejuvenation steps performed for each resampled theta particle.
            threshold (float): ESS (Effective Sample Size) threshold (as a proportion of N_theta or N_x)
                               that triggers resampling/rejuvenation.
        """
        self.p = p # number of lag for the VAR model [cite: 8]
        self.N_theta = N_theta # number of theta particles (parameter particles)
        self.N_x = N_x # number of inner particles for each theta particle
        self.threshold = threshold # ESS threshold (proportion) to trigger resampling/rejuvenation
        self.N_rejuvenate = N_rejuvenate # number of rejuvenation steps per resampled particle
        self.logfile = logfile # path to the log file
        
        # M_theta: For waste-free SMC, this defines the number of particles to be resampled
        # such that M_theta * (N_rejuvenation + 1) = N_theta
        self.M_theta = N_theta // (N_rejuvenate + 1)
        
        self.mem_usage =[] # log to track memory usage during execution

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
        
        # trajectories: Stores the log-lambda trajectories for each inner particle.
        # Shape (N_theta, N_x, K, T+1). The last dimension (T+1) includes the initial
        # log-lambda_0 (at index 0) and T time steps of log-lambda_t (at indices 1 to T).
        self.trajectories = np.zeros((self.N_theta, self.N_x, self.K, self.T + 1))
        
        # ancestors: Stores ancestor indices for the inner CSMC particles.
        # Shape (N_theta, N_x, K, T). ancestor[:,:,:,t-1] refers to indices for trajectories[:,:,:,t].
        self.ancestors = np.zeros((self.N_theta, self.N_x, self.K, self.T))
        
        # Initialize ancestor indices for the very first step (t=0) to be simply 0 to N_x-1,
        # implying no resampling at initialization.
        for i in range(self.N_theta):
            for k in range(self.K):
                self.ancestors[i, :, k, 0] = np.arange(self.N_x)
        
        # Initialize storage for parameter particles (theta particles)
        # A: Lower triangular matrix with ones on the diagonal. Shape (N_theta, K, K).
        self.A = np.zeros((self.N_theta, self.K, self.K))
        # Pi: Regression coefficients matrix. Shape (N_theta, K, K*p + 1). 
        self.Pi = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        # Api: Product of A and Pi, pre-calculated for efficiency. Shape (N_theta, K, K*p + 1).
        self.Api = np.zeros((self.N_theta, self.K, self.K * self.p + 1))
        # Phi: Diagonal elements of the Phi matrix (variance of log-lambda increments). Shape (N_theta, K).
        self.Phi = np.zeros((self.N_theta, self.K))

    def initialize_from_prior(self, y):
        """
        Initializes the parameter particles (A, Pi, Phi) and the initial log-lambda trajectories
        by drawing samples from their respective prior distributions, as outlined in the algorithm document.
        """

        # Sample A matrices for each theta particle (N_theta particles)
        # Prior for A is non-informative independent Gaussian on each row 
        # A is a lower triangular matrix with ones on the diagonal 
        for i in range(self.N_theta):
            for j in range(self.K): # Iterate through rows of A
                if j > 0:
                    # For rows j > 0, sample the first j-1 elements from the specified Gaussian prior.
                    # A_j,1:j-1 ~ N(0, I*10^6) 
                    self.A[i, j, :j] = np.random.multivariate_normal(self.mu_A[:j], self.Sigma_A[:j, :j])
                # Set the diagonal element to 1 
                self.A[i, j, j] = 1.0

        # Sample Pi matrices for each theta particle
        # Pi is initialized by drawing from its Minnesota prior
        for i in range(self.N_theta):
            for j in range(self.K): # Iterate through rows of Pi
                # Each row Pi^(j) ~ N(mu_Pi^(j), Sigma_Pi^(j)) [cite: 3, 46]
                self.Pi[i, j, :] = np.random.multivariate_normal(
                    self.Pi_prior_mean[j], self.Pi_prior_var_rows[j]
                )

        # Compute Api (A @ Pi) for each theta particle
        # This is a pre-calculation for efficiency in subsequent steps.
        for i in range(self.N_theta):
            self.Api[i] = self.A[i] @ self.Pi[i]

        # Sample Phi (diagonal elements) for each theta particle
        # Prior for Phi_j,j is Inverse Gamma((K+2)/2, 1/2) 
        alpha = (self.K + 2) / 2 # Shape parameter for Inverse Gamma prior 
        beta = 1 / 2 # Scale parameter for Inverse Gamma prior (note: np.random.gamma uses scale=1/beta for InvGamma)
        # Sample from Gamma distribution and then invert to get Inverse Gamma samples
        phi_samples = np.random.gamma(shape=alpha, scale=1.0, size=(self.N_theta, self.K))
        self.Phi = 1.0 / (phi_samples * beta)

        # Sample log-lambdas at time 0 for all inner particles of all theta particles
        # Prior for log_lambda_0,j ~ N(0, sigma_0) 
        self.trajectories[:, :, :, 0] = np.random.normal(
            loc=0.0,
            scale= np.sqrt(self.sigma_0), # sigma_0 is the variance from init_prior 
            size=(self.N_theta, self.N_x, self.K)
        )

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
            for j in range(self.N_rejuvenation + 1):
                target_index = i * (self.N_rejuvenation + 1) + j
                # Assign sampled A, Pi, Phi parameters to the corresponding theta particles
                self.A[target_index] = sample_result['A'][j]
                self.Pi[target_index] = sample_result['Pi'][j]
                self.Phi[target_index] = sample_result['Phi'][j]
                # Recompute Api for the newly assigned A and Pi
                self.Api[target_index] = self.A[target_index] @ self.Pi[target_index]
            
            # Assign the sampled log-lambda trajectories and ancestor indices.
            # Note: The 'trajectories' are initialized up to 'init_t + 1' (including lambda_0 to lambda_init_t),
            # and 'ancestors' up to 'init_t' (for steps from 0 to init_t-1).
            self.trajectories[i*(self.N_rejuvenation + 1) : (i+1)*(self.N_rejuvenation + 1),:,:, 0:init_t + 1] = sample_result['trajectories']
            self.ancestors[i*(self.N_rejuvenation + 1) : (i+1)*(self.N_rejuvenation + 1),:,:, 0:init_t] = sample_result['ancestors']

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
        self.trajectories[:self.M_theta] = self.trajectories[indices]
        self.ancestors[:self.M_theta] = self.ancestors[indices]
        
        # Outer weights are reset to zero (on log scale) after resampling,
        # as these resampled particles are now considered to have equal initial weight
        # for the next cycle of importance sampling.
        self.outer_weights = np.zeros(self.N_theta)
        
        # Inner weights for the resampled particles are also reassigned,
        # maintaining the relative weights within each inner CSMC chain.
        self.inner_weights[:self.M_theta] = self.inner_weights[indices]

     def safe_replace_after_rejuvenation(self):
        """
        Implements a safeguard mechanism to detect and replace 'bad' (NaN/Inf) particles
        after the rejuvenation step. This is crucial for numerical stability and
        to prevent divergence of the SMC chains, ensuring that all N_theta particles
        remain valid for subsequent iterations.
        """
        # Identify 'bad' particles across all relevant attributes (A, Pi, Phi, trajectories).
        # A particle is considered 'bad' if any of its components (A, Pi, Phi, or any log-lambda trajectory point)
        # contains NaN (Not a Number) or Inf (Infinity) values.
        is_bad = (
            np.any(np.isnan(self.A), axis=(1, 2)) | # Check for NaN in A matrix
            np.any(np.isnan(self.Pi), axis=(1, 2)) | # Check for NaN in Pi matrix
            np.any(np.isnan(self.Phi), axis=1) | # Check for NaN in Phi vector
            np.any(np.isnan(self.trajectories), axis=(1, 2, 3)) # Check for NaN in log-lambda trajectories
        )
        # Note: np.any also implicitly handles Inf if the operation results in NaN,
        # but a more robust check might explicitly include np.isinf as well if non-NaN Inf is possible.

        num_bad = np.sum(is_bad) # Count the total number of bad particles.
        if num_bad == 0:
            return  # If no bad particles are found, exit the function.

        self.log(f"[SafeGuard] Detected {int(num_bad)} NaN/Inf particles after rejuvenation — replacing.")

        # Prepare weights for resampling *valid* particles to replace the bad ones.
        valid_mask = ~is_bad # Create a boolean mask to identify valid particles.
        weights = self.outer_weights.copy() # Use a copy of current outer weights.
        
        # Set weights of invalid particles to -inf to ensure they are not chosen for replacement.
        # This handles potential NaN/Inf in weights themselves.
        weights[~np.isfinite(weights)] = -np.inf 
        weights[is_bad] = -np.inf 

        # Calculate stable probabilities for valid particles using log-sum-exp trick.
        # If no valid particles exist, raise an error as replacement is impossible.
        max_weight = np.max(weights[valid_mask]) if np.any(valid_mask) else -np.inf
        stable_weights = np.exp(weights - max_weight)
        
        # Ensure sum of stable_weights is not zero to prevent division by zero, if no valid particles.
        sum_stable_weights = np.sum(stable_weights)
        if sum_stable_weights == 0:
            raise RuntimeError("All particles are invalid — safe replacement not possible.")
        
        probs = stable_weights / sum_stable_weights # Normalize to get probabilities.

        # Get indices of the valid particles.
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise RuntimeError("All particles are invalid — safe replacement not possible.")

        # Sample replacement indices from the valid particles with their probabilities.
        replacement_indices = np.random.choice(valid_indices, size=num_bad, p=probs[valid_indices])

        # Get indices of the bad particles.
        bad_indices = np.where(is_bad)[0]

        # Replace the attributes of the bad particles with those of the randomly sampled valid particles.
        self.A[bad_indices] = self.A[replacement_indices]
        self.Pi[bad_indices] = self.Pi[replacement_indices]
        self.Api[bad_indices] = self.Api[replacement_indices]
        self.Phi[bad_indices] = self.Phi[replacement_indices]
        self.trajectories[bad_indices] = self.trajectories[replacement_indices]
        self.ancestors[bad_indices] = self.ancestors[replacement_indices]
        self.inner_weights[bad_indices] = self.inner_weights[replacement_indices]
        
        # Optionally, set outer weights of replaced particles to -inf (on log scale)
        # to ensure they don't contribute significantly until the next proper resampling.
        self.outer_weights[bad_indices] = -np.inf # Keep weight down until next proper resample

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
        logw -= scipy.special.logsumexp(logw, axis=0, keepdims=True)
        weights = np.exp(logw) # Convert log weights to probabilities for sampling.
        
        # Initialize storage for the K sampled log-lambda trajectories.
        # These fixed trajectories are used for rejuvenating Pi parameters.
        # 'np.inf' is used as a placeholder for unobserved/future log-lambdas ang y to effectively
        # zero out their contribution in certain calculations.
        fix_log_lambda_i = np.full((self.K, self.current_t + 1), np.inf)
        
        # Initialize list for K sampled trajectories specifically for Phi rejuvenation.
        # Posterior for Phi_k is independent for each component, and 'compute_posterior_phi_by_component'
        # requires a list of sampled trajectories with potentially different lengths.
        fix_log_lambda_phi_i = []
        df_phi = [] # List to store degrees of freedom for each Phi component's posterior.
        
        # Sample backward for the K initial fixed trajectories (one for each k component).
        # This fixes a path of log-lambdas for use in the subsequent Gibbs updates for A, Pi, Phi.
        for k in range(self.K): # Iterate over each of the K variables/components.
            # Sample an index for the fixed trajectory from the inner particles based on their weights.
            idx = np.random.choice(self.N_x, p=weights[:, k])
            if k <= self.current_k:
                # For components 'k' that have already been observed up to 'current_t':
                # The degree of freedom for Phi_k posterior is (current_t + 1) because
                # log-lambdas are observed from t=0 to t=current_t.
                df_phi.append(self.current_t + 1)
                # Backward sample the trajectory from t=current_t down to t=0.
                # Note: self.trajectories[:, :, k, t+1] stores log_lambda_t values.
                for t_rev in reversed(range(self.current_t + 1)): # Iterate from current_t down to 0.
                    fix_log_lambda_i[k, t_rev] = self.trajectories[i, idx, k, t_rev + 1]
                    idx = self.ancestors[i, idx, k, t_rev].astype(int) # Move to ancestor index.
                # Include onlt the observed log-lambdas for rejuvenation of Phi.
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :])
            else:
                # For components 'k' that are unobserved at 'current_t' (i.e., k > current_k):
                # We only have observations up to 'current_t - 1'.
                # The degree of freedom for Phi_k posterior is 'current_t'.
                df_phi.append(self.current_t)
                # Backward sample the trajectory from t=current_t-1 down to t=0.
                for t_rev in reversed(range(self.current_t)): # Iterate from current_t-1 down to 0.
                    fix_log_lambda_i[k, t_rev] = self.trajectories[i, idx, k, t_rev + 1]
                    idx = self.ancestors[i, idx, k, t_rev].astype(int)
                # Include only the observed log-lambdas for rejuvenation of Phi.
                fix_log_lambda_phi_i.append(fix_log_lambda_i[k, :self.current_t])
        
        # Initialize the parameters for this specific theta particle from its current state.
        A_i = self.A[i].copy()
        Pi_i = self.Pi[i].copy()
        Phi_i = self.Phi[i].copy()
        Ay = self.y @ A_i.T 

        # Perform N_rejuvenation steps of Gibbs sampling.
        for j in range(self.N_rejuvenate):
            # Determine the target index where the rejuvenated particle's parameters will be stored.
            # This is part of the waste-free scheme, where rejuvenated particles fill new slots.
            target_idx = self.M_theta + i * self.N_rejuvenation + j

            # Rejuvenation of Pi 
            # Updates Pi_i conditioned on A_i, fixed log-lambda paths, y, and z.
            # Refer to Section 3 "Pi", specifically Eq. (59) and (60) for posterior mean and covariance.
            # Note: The `update_APi` function is assumed to implement this update logic.
            # We pass in y and z till time current_t
            # for the k such that y_{current_t,k} unobserved, fix_log_lambda_i at the correspond position
            # is np.inf, which will cancel out and contribute 0
            Pi_i = update_APi(
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
            # Note: The `compute_posterior_phi_by_component` function is assumed to implement this.
            Phi_i = compute_posterior_phi_by_component(
                fix_log_lambda_phi_i, # Sampled log-lambda paths for Phi rejuvenation.
                df_phi, # Degrees of freedom for each Phi component.
                K=self.K # Number of variables.
            )
            
            # Update Api (A @ Pi) with the newly sampled Pi_i.
            Api_i = A_i @ Pi_i
            
            # Empty the list for Phi rejuvenation's fixed log lambdas, as they will be re-sampled.
            fix_log_lambda_phi_i = []

            # Rejuvenation of log-lambda trajectories using Conditional Sequential Monte Carlo (CSMC)
            # This loops through each of the K components, running a CSMC for each.
            for k in range(self.K):
                # Determine the observation time window for CSMC based on whether 'k' has been observed at current_t.
                t_k = self.current_t + 1 if k <= self.current_k else self.current_t
                
                # Initialize CSMC for the k-th component.
                # 'fixed_particles' ensures the path used in previous Gibbs steps is maintained as the first particle.
                csmc = CSMC(
                    Num=self.N_x, # Number of inner particles for CSMC.
                    phi=Phi_i, # Current Phi_i for this particle.
                    sigma0=self.sigma_0, # Initial variance for log-lambda.
                    y=Ay[:t_k], # Observed Ay up to t_k.
                    z=self.z[:t_k], # Observed z up to t_k.
                    B=Api_i, # Current Api_i for this particle.
                    j=k, # The specific component k.
                    fixed_particles=fix_log_lambda_i[k, :t_k] # The fixed log-lambda path for this k.
                )
                
                # Run CSMC and update trajectories and ancestors.
                # sampled_traj is the (new) fixed path for this k, reconstructed from CSMC's backward sampling.
                self.trajectories[target_idx, :, k, 1:t_k + 1], \
                self.ancestors[target_idx, :, k, :t_k], \
                sampled_traj, _, _ = csmc.run(if_reconstruct=True)
                
                # Update fixed log-lambda and add to the list for Phi rejuvenation.
                fix_log_lambda_phi_i.append(sampled_traj)
                fix_log_lambda_i[k, :t_k] = sampled_traj # Update the fixed path used for A/Pi rejuvenation.
            
            # Rejuvenation of A (Lower triangular matrix)
            # Refer to Section 2 "A", specifically Eq. (26) and (27) for posterior mean and covariance.
            
            # For unobserved y_{current_t, k} (where k > current_k), we need to simulate them
            # to complete the y matrix for the A update.
            # Calculate A_i @ Pi_i @ z_t for the current time step.
            APiz = A_i @ Pi_i @ self.z[self.current_t]
            for k_unobs in range(self.current_k + 1, self.K):
                # Simulate future log-lambda at (current_t, k_unobs) based on random walk.
                fix_log_lambda_i[k_unobs, self.current_t] = fix_log_lambda_i[k_unobs, self.current_t - 1] + np.sqrt(Phi_i[k_unobs]) * np.random.normal(0,1)
                # Simulate Ay at (current_t, k_unobs) based on the simulated log-lambda.
                # Assuming (Ay_{t,k} - APiz_t)^2 / lambda_{t,k} follows a Normal distribution.
                Ay[self.current_t, k_unobs] = np.random.normal(
                    APiz[k_unobs], # Mean is the deterministic part.
                    np.exp(fix_log_lambda_i[k_unobs, self.current_t]) ** .5 # Std dev is sqrt(lambda_t,k).
                )
            
            # Construct a 'synthetic_y' matrix for the A update,
            # which combines observed y (up to current_t, current_k) with simulated y (at current_t, for k > current_k).
            # Note: A_i.T is (K,K), np.linalg.inv(A_i.T) is (K,K). This reconstructs y from Ay.
            syn_y = Ay[:self.current_t + 1] @ np.linalg.inv(A_i.T)
            
            # Now, update A_i using the synthetic_y matrix.
            # Note: The `compute_posterior_A_with_log_lambdas` function is assumed to implement this.
            A_i = compute_posterior_A_with_log_lambdas(
                syn_y, # The (partially simulated) y matrix.
                self.z[:self.current_t + 1], # z up to current_t.
                fix_log_lambda_i, # Fixed log-lambda trajectories.
                self.mu_A, self.Sigma_A, # Priors for A.
                Pi_i # Current Pi_i.
            )
            
            # After A is updated, recompute Ay using the real observed data (self.y)
            # and the newly sampled A_i. This discards any simulated y values for consistency.
            Ay = self.y @ A_i.T
            
            # Discard the predicted log_lambda_{current_t, current_k+1:K} and reinitialize them as np.inf.
            # This prepares 'fix_log_lambda_i' for the next rejuvenation step for Pi.
            fix_log_lambda_i[self.current_k + 1 : self.K , self.current_t] = np.full((self.K - self.current_k - 1, ), np.inf)
            
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
        
        # Use multiprocessing.Manager to create a shared list 'tracker' that can be accessed
        # by multiple parallel processes/threads.
        with Manager() as manager:
            tracker = manager.list() # Shared list to track worker IDs.
            
            # Execute '_rejuvenate_one_theta' in parallel for each of the M_theta particles.
            # n_jobs=-1 uses all available CPU cores/threads.
            # backend="threading" is chosen, indicating thread-based parallelism.
            Parallel(n_jobs=-1, backend="threading")(
                delayed(self._rejuvenate_one_theta)(i, rejuvenate_A, tracker) # Call _rejuvenate_one_theta for each index 'i'
                for i in range(self.M_theta) # Iterate through the M_theta particles to be rejuvenated
            )

            # Log the number of unique parallel workers used during rejuvenation.
            unique_workers = set(tracker)
            self.log(f"[✓] Rejuvenation complete. Used {len(unique_workers)} parallel workers.")
            
        # After all rejuvenation steps are complete, call the safeguard function.
        # This checks for and replaces any NaN/Inf particles that might have resulted
        # from numerical instabilities during the rejuvenation process, ensuring robustness.
        self.safe_replace_after_rejuvenation()

    def run(self, y, t_init=0, init_result=None, if_pred=False):
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
        """
        
        # Set up prior values and initialize parameters for the model.
        # This populates self.A, self.Pi, self.Phi, self.trajectories, etc., based on priors.
        self.init_prior(y)

        # Initialize tracking lists for algorithm diagnostics.
        self.ESS_track = []         # Tracks Effective Sample Size at each step.
        self.likelihood_track = []  # Tracks marginal log-likelihood at each step.
        self.predicted_residuals = [] # Stores predicted residuals if if_pred is True.
        
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
                if ess_k < self.threshold * self.N_theta:
                    self.rejuvenate_parallelized()
                
                # Track memory usage at each (t,k) step.
                current, peak = tracemalloc.get_traced_memory()
                self.mem_usage.append(current / 1e6) # Log in MB.

            self.log(f'final ess {ess_k}') # Log final ESS for the time step 't'.
            
            # Optional: Perform prediction for future residuals.
            if if_pred:
                pred = self.monte_carlo_predictive_residuals()
                self.log(f'at time t = {t}, monte_carlo_predictive_residuals = {pred}')
                self.predicted_residuals.append(pred)

    def monte_carlo_predictive_residuals(self, num_samples=1000):
        """
        Computes Monte Carlo predictive residuals based on the current set of
        outer (theta) particles and their corresponding log-lambda trajectories.
        This provides a distribution of potential future residuals from the VAR model.
        
        Args:
            num_samples (int): The number of samples to draw for Monte Carlo prediction.
        
        Returns:
            np.ndarray: An array containing the lower 2.5 percentile, upper 97.5 percentile,
                        and mean of the predicted residuals across each dimension (K).
                        Shape: (3, K).
        """
        # Normalize outer weights to get probabilities for sampling theta particles.
        probs = np.exp(self.outer_weights - np.max(self.outer_weights))
        probs /= np.sum(probs)

        # Sample 'num_samples' indices of theta particles based on their probabilities.
        # These selected particles will be used to generate predictive residuals.
        indices = np.random.choice(self.N_theta, size=num_samples, p=probs)
        predicted_residuals = []

        # Generate a predictive residual for each sampled theta particle.
        for i in range(num_samples):
            idx = indices[i] # Get the index of the sampled theta particle.
            A_i = self.A[idx] # Retrieve the A matrix for this particle.
            
            # Compute the inverse of A_i. This is needed because v_t = A^{-1} Lambda_t^{1/2} epsilon_t,
            # and we are interested in v_t (the residual).
            A_i_inv = np.linalg.inv(A_i) 
            
            # Pi_i is not directly used in the residual calculation but might be relevant
            # for the full VAR equation (y_t - Pi z_t).
            Pi_i = self.Pi[idx] 

            # Randomly select one inner particle's log-lambda trajectory for each of the K components
            # at the current time step 'self.current_t'.
            idxs = np.random.randint(0, self.N_x, size=self.K)
            fix_log_lambda_i = self.trajectories[idx, idxs, np.arange(self.K), self.current_t + 1] # self.current_t + 1 accesses lambda_t
            
            # Construct the diagonal Lambda_t matrix from the sampled log-lambdas.
            # Lambda_t = diag(exp(log_lambda_t)).
            Lambda_i = np.diag(np.exp(fix_log_lambda_i))
            
            # Compute the covariance matrix for the residual (v_t).
            # From v_t = A^{-1} Lambda_t^{1/2} epsilon_t, where epsilon_t ~ N(0, I_K),
            # the covariance of v_t is Cov(v_t) = A^{-1} Lambda_t (A^{-1})^T.
            Sigma_i = A_i_inv @ Lambda_i @ A_i_inv.T

            try:
                # Sample a residual from a multivariate normal distribution with mean 0 and computed covariance.
                residual_i = np.random.multivariate_normal(mean=np.zeros(self.K), cov=Sigma_i)
            except np.linalg.LinAlgError:
                # Handle cases where Sigma_i might not be Positive Definite (PD) due to numerical issues.
                # In such cases, log a warning and fall back to sampling from a small noise distribution.
                self.log(f"[Warning] Non PD covariance at sample {i}, fallback to small noise.")
                residual_i = np.random.normal(0, 1e-4, size=self.K) # Fallback to very small noise.

            predicted_residuals.append(residual_i)

        predicted_residuals = np.array(predicted_residuals) # Convert list of residuals to numpy array.
        
        # Calculate summary statistics for the predicted residuals: lower 2.5 percentile,
        # upper 97.5 percentile (for 95% credible interval), and the mean.
        lower = np.percentile(predicted_residuals, 2.5, axis=0)
        upper = np.percentile(predicted_residuals, 97.5, axis=0)
        mean_residual = np.mean(predicted_residuals, axis=0)

        # Return the summary statistics as a single array.
        return np.array([lower, upper, mean_residual])
    
    def _smc_inner_loop(self, i, t, k, Phi_k, A_k, Api_k, y_t, z_t):
        """
        Performs a single step of the Conditional Sequential Monte Carlo (CSMC) algorithm
        for the k-th component of the i-th theta particle at time step t.
        This involves proposing new log-lambda particles, updating their weights,
        and performing resampling if the Effective Sample Size (ESS) falls below a threshold.
        This function corresponds to Algorithm 1 (CSMC) in the attached document,
        executed independently for each (theta_i, k) pair.
        
        Args:
            i (int): Index of the outer (theta) particle.
            t (int): Current time step (from 0 to T-1).
            k (int): Current variable component index (from 0 to K-1).
            Phi_k (np.ndarray): The Phi parameter for the k-th component,
                                extracted from all N_theta particles (shape: (N_theta, 1)).
            A_k (np.ndarray): The k-th row of the A matrix for all N_theta particles (shape: (N_theta, 1, K)).
            Api_k (np.ndarray): The k-th row of the A@Pi product for all N_theta particles (shape: (N_theta, 1, Kp+1)).
            y_t (np.ndarray): The observed y vector at time t (shape: (K, 1)).
            z_t (np.ndarray): The z vector at time t (shape: (Kp+1, 1)).
            
        Returns:
            float: The log-likelihood contribution for y_{t,k} for the i-th particle.
        """
        # Retrieve previous log-lambda particles based on current time step.
        if t == 0:
            # At t=0, particles are taken directly from their initialization (prior samples).
            log_lambda_prev = self.trajectories[i, :, k, t]
        else:
            # For t > 0, retrieve ancestor indices from the previous time step (t-1).
            idx = self.ancestors[i, :, k, t-1].astype(int)
            # Retrieve the inner particles of (t-1,k) based on ancestor indices.
            log_lambda_prev = self.trajectories[i, idx, k, t] # self.trajectories[:,:,:,t] holds lambda_{t-1} values for t>0

        # Step forward: Propose new log-lambda particles at (t, k) using a bootstrap proposal.
        # Proposal: log_lambda_t,j^(n) ~ N(log_lambda_t-1,j^(a_t^n), Phi_j,j).
        log_lambda_t = log_lambda_prev + np.sqrt(Phi_k[i]) * np.random.normal(size=self.N_x)
        
        # Clip log_lambda_t values to prevent numerical instabilities from extreme values.
        log_lambda_t = np.clip(log_lambda_t, a_min=-10.0, a_max=10.0)
        
        # Record the proposed particle cloud for the current time step.
        # self.trajectories[:,:,:,t+1] stores log_lambda_t values.
        self.trajectories[i, :, k, t + 1] = log_lambda_t
        
        # Calculate the residual term for the likelihood: (Ay_{t,k} - APiz_t)^2.
        # This is derived from the likelihood form in Eq. (72) for log-lambda.
        # y_t is (K,1), z_t is (Kp+1,1)
        A = A_k[i].reshape(1, -1) # Reshape A_k for matrix multiplication: (1, K)
        Api = Api_k[i].reshape(1, -1) # Reshape Api_k for matrix multiplication: (1, Kp+1)
        
        # Compute the specific component of the residual for the j-th row.
        # (A_j,. @ y_t - (A@Pi)_j,. @ z_t)
        res = (A @ y_t - Api @ z_t).squeeze()
        res = np.clip(res, -1e3, 1e3) # Clip residual to prevent extreme values.
        
        # Compute the unnormalized log weight for each inner particle.
        # Here, the weight is  the likelihood term -0.5 * (log(lambda_t,j) + (Ay_{t,k} - Pi_j z_t)^2 / lambda_{t,k}).
        lw = -0.5 * (np.log(2 * np.pi) + log_lambda_t + np.exp(-log_lambda_t) * res**2)
        
        # Handle cases where log weight might become NaN (e.g., from log(0) or inf/nan inputs).
        lw = np.where(np.isnan(lw), -np.inf, lw)
        
        # Update the inner weight for each particle by adding the new log likelihood term.
        self.inner_weights[i, :, k] += lw
        
        # Compute the log likelihood contribution for y_{t,k} for the i-th theta particle.
        # This is the estimated marginal likelihood for the observation (y_t,k) given the theta particle.
        # Equivalent to log(sum(exp(lw_n))) - log(N_x)
        ll_i = scipy.special.logsumexp(lw) - np.log(self.N_x)

        # Normalize the inner weights for ESS calculation and resampling.
        self.inner_weights[i, :, k] -= scipy.special.logsumexp(self.inner_weights[i, :, k])
        weights = np.exp(self.inner_weights[i, :, k])
        # A small safeguard against zero sum weights
        weights /= np.sum(weights) if np.sum(weights) > 0 else self.N_x
        
        # Compute Effective Sample Size (ESS).
        # ESS = 1 / sum(normalized_weights^2)
        ess = 1.0 / np.sum(weights ** 2)
        
        # Resampling step: If ESS is below threshold, perform multinomial resampling.
        if ess < self.threshold * self.N_x: # Using N_x as the base for inner ESS threshold.
            # Sample new ancestor indices based on the normalized inner weights.
            new_idx = np.random.choice(self.N_x, size=self.N_x, p=weights)
            self.ancestors[i, :, k, t] = new_idx
            # Reset inner weights to zero (on log scale) after resampling.
            self.inner_weights[i, :, k] = np.zeros(self.N_x)
        else:
            # If ESS is sufficient, set ancestor indices to identity (no resampling).
            self.ancestors[i, :, k, t] = np.arange(self.N_x)
        
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