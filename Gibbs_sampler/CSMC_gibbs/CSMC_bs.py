import numpy as np

class CSMC:
    """
    Implements a Conditional Sequential Monte Carlo (CSMC) filter.
    
    This algorithm is also known as a Conditional Particle Filter. It is used to sample
    trajectories from the distribution of a state-space model, conditioned on the fact 
    that one of the particles must follow a predefined path. This is a key component of 
    the Particle Gibbs (PG) algorithm.
    """
    
    def __init__(self, Num, phi, sigma0, y, z, B, j, fixed_particles, ESSmin=0.5):
        """
        Initializes the CSMC filter with model parameters, data, and the conditioning trajectory.

        Args:
            Num (int): The total number of particles.
            phi (float): The variance for the state transition (random walk).
            sigma0 (float): The variance for the prior distribution of the initial state x_{j,0}.
            y (np.ndarray): The observations array, shape (T, N).
            z (np.ndarray): An auxiliary variable array, shape (T, Np+1).
            B (np.ndarray): A matrix used in the observation model (B = A @ Pi).
            j (int): The specific row index in the observation/parameter arrays to use.
            fixed_particles (np.ndarray): The predefined trajectory that one particle is forced to follow.
            ESSmin (float): The minimum Effective Sample Size (ESS) threshold, as a fraction of Num,
                            below which resampling is triggered. Defaults to 0.5.
        """
        # --- Model and Data Parameters ---
        self.Num = Num  # Total number of particles
        self.sigma0 = sigma0  # Variance for the prior of x_{j,0}
        self.phi = phi  # Variance for the random walk process noise
        self.y = y  # Observations, shape (T, N)
        self.z = z  # Auxiliary data, shape (T, Np+1)
        self.B = B  # Matrix B = A@Pi
        self.j = j  # Row index for the current process
        self.fixed_particles = fixed_particles  # The trajectory to condition on
        self.ESSmin = ESSmin  # ESS threshold for adaptive resampling
        self.T = self.y.shape[0] # Total number of time steps

        # --- Internal State and History ---
        # Stores the full paths of all particles. Shape: (Num, T)
        self.trajectories = np.zeros((Num, len(fixed_particles)))
        # Stores the ancestor index for each particle at each time step after resampling.
        self.ancestors = np.zeros((Num, len(fixed_particles)), dtype=int)
        # Stores the cumulative log weights of the particles.
        self.weights = np.zeros(self.Num)
        
    def initialize_particles(self):
        """
        Initializes particles at t=0 from the prior and computes their initial weights.
        The conditioning particle's value is set from the fixed trajectory.
        """
        self.particles = np.random.normal(0., self.sigma0**0.5, size=self.Num)
        self.weights = self.compute_log_weights(self.particles, 0)

    def compute_log_weights(self, particles, t):
        """
        Computes the log-likelihood (log-weights) for each particle in a vectorized manner.

        Args:
            particles (np.ndarray): The array of current particle states.
            t (int): The current time step.

        Returns:
            np.ndarray: An array of log-weights for the particles.
        """
        # Calculate the residual (innovation) term for the observation model.
        residuals = self.y[t, self.j] - np.dot(self.B[self.j, :], self.z[t, :])
        residuals = np.clip(residuals, -1e3, 1e3) # Clip for numerical stability.
        
        # Compute log-weights based on the model's specific likelihood function.
        log_weights = -0.5 * (particles + np.exp(-particles) * (residuals**2))
        return log_weights

    def resample(self, t):
        """
        Performs multinomial resampling if the ESS is below the threshold.
        Crucially, it forces the first particle's ancestor to be itself, preserving the fixed trajectory.
        
        Args:
            t (int): The current time step.

        Returns:
            np.ndarray: The array of ancestor indices for each particle.
        """
        # Convert log-weights to normalized weights using the log-sum-exp trick for stability.
        max_log_weight = np.max(self.weights)
        weights = np.exp(self.weights - max_log_weight)
        weights /= weights.sum()

        # Perform resampling using the inverse CDF method to get ancestor indices.
        indices = np.searchsorted(np.cumsum(weights), np.random.rand(self.Num))
        
        # --- This is the key step for Conditional SMC ---
        # Force the first particle's ancestor to be itself (index 0).
        # This ensures the lineage of the conditioned particle is preserved.
        indices[0] = 0

        # Resample the particles based on the new indices.
        self.particles = self.particles[indices]
        
        # Store the ancestor indices.
        self.ancestors[:, t] = indices
        
        # Reset weights to zero after resampling. In a log-weight filter, this is
        # equivalent to setting all weights to 1/N.
        self.weights = np.zeros(self.Num)
        
        return indices

    def update_particles(self, t):
        """
        Propagates (updates) particles according to the dynamic model.
        The first particle is clamped to the value from the fixed trajectory.
        
        Args:
            t (int): The current time step.
        """
        # Propagate all particles (except the first) with random walk noise.
        noise = np.random.normal(0, self.phi[self.j]**0.5, size=self.Num)
        self.particles[1:] += noise[1:]
        
        # --- This is the key step for Conditional SMC ---
        # Force the first particle to follow the predefined trajectory.
        self.particles[0] = self.fixed_particles[t]
        
        # Clip particles to prevent divergence.
        self.particles = np.clip(self.particles, -10, 10)
        
        # Store the updated particle values in the trajectory history.
        self.trajectories[:, t] = self.particles

    def backward_sampling(self):
        """
        Performs backward sampling to draw a single, coherent trajectory from the
        entire history of particles and their weights. This is an alternative to
        simply tracing back ancestors.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The single sampled trajectory.
                - np.ndarray: The ancestor indices corresponding to the sampled trajectory.
        """
        sampled_trajectory = np.zeros(self.T)
        sampled_ancestor = np.zeros(self.T, dtype=int)

        # 1. Sample the final particle at time T-1 from the final weighted distribution.
        log_w = self.compute_log_weights(self.particles, self.T - 1)
        weights = np.exp(log_w - np.max(log_w))
        weights /= weights.sum()
        sampled_index = np.searchsorted(np.cumsum(weights), np.random.rand(1))[0]
        sampled_trajectory[-1] = self.trajectories[sampled_index, -1]
        sampled_ancestor[-1] = sampled_index

        # 2. Iterate backward from T-2 to 0.
        for t in range(self.T - 2, -1, -1):
            # Get particles and their original weights at time t.
            particles_t = self.trajectories[:, t]
            log_weights_t = self.compute_log_weights(particles_t, t)
            
            # Get the already-sampled particle at the next time step.
            particle_next = sampled_trajectory[t + 1]
            
            # Calculate the transition probability from each particle at t to the chosen particle at t+1.
            # Then, update the weights with this information (Bayes' rule).
            log_transition_prob = -(particle_next - particles_t)**2 / (2 * self.phi[self.j])
            log_weights_tilde = log_weights_t + log_transition_prob
            
            # Normalize the new weights and sample an ancestor.
            weights_tilde = np.exp(log_weights_tilde - np.max(log_weights_tilde))
            weights_tilde /= weights_tilde.sum()
            sampled_index = np.searchsorted(np.cumsum(weights_tilde), np.random.rand(1))[0]
            
            sampled_trajectory[t] = particles_t[sampled_index]
            sampled_ancestor[t] = sampled_index
            
        return sampled_trajectory, sampled_ancestor

    def compute_ESS(self):
        """
        Computes the Effective Sample Size (ESS).
        ESS is a measure of particle degeneracy. A value close to `Num` is good,
        while a value close to 1 is bad.
        """
        # Use log-sum-exp trick for numerical stability.
        max_log_weight = np.max(self.weights)
        weights = np.exp(self.weights - max_log_weight)
        weights /= weights.sum()
        ESS = 1. / np.sum(weights**2)
        return ESS

    def run(self, if_reconstruct=False):
        """
        Executes the main CSMC filter loop.

        Args:
            if_reconstruct (bool): If True, performs a final resampling step to ensure
                                   trajectories are equally weighted for reconstruction.

        Returns:
            tuple: A tuple containing trajectories, ancestors, a backward-sampled path,
                   and final weights.
        """
        # 1. Initialize particles and their weights at t=0.
        self.initialize_particles()

        # 2. Main loop: Iterate through time steps t=0 to T-1.
        for t in range(self.T):
            # Set default ancestor to self before resampling.
            self.ancestors[:, t] = np.arange(self.Num)
            
            # Check ESS and resample if it falls below the threshold.
            if self.compute_ESS() < self.ESSmin * self.Num:
                indices = self.resample(t)

            # 3. Propagate particles to the next state.
            self.update_particles(t)

            # 4. Compute new log-weights and add them to the cumulative weights.
            log_weights = self.compute_log_weights(self.particles, t)
            self.weights += log_weights
            
        # Optional final resampling step (for SMC Square)
        if if_reconstruct:
            indices = self.resample(self.T - 1)

        # 5. Perform backward sampling to get a single path.
        sampled_trajectory, sampled_ancestor = self.backward_sampling()

        return self.trajectories, self.ancestors, sampled_trajectory, sampled_ancestor, self.weights

    def reconstruct_path(self, trajectories, ancestors, final_indices):
        """
        Reconstructs particle paths by tracing backward through the stored ancestor indices.

        Args:
            trajectories (np.ndarray): The full history of particle values [Num, T].
            ancestors (np.ndarray): The history of ancestor indices [Num, T].
            final_indices (np.ndarray): The indices of the particles at the final time step
                                        from which to start tracing back.

        Returns:
            np.ndarray: The set of reconstructed paths [T, N].
        """
        N, T = trajectories.shape
        paths = np.zeros((T, N))

        # For each particle at the final time...
        for n in range(N):
            # ...start with its given final index.
            idx = final_indices[n]
            # Trace back its lineage to t=0.
            for t in reversed(range(T)):
                paths[t, n] = trajectories[idx, t]
                idx = ancestors[idx, t] # Move to the ancestor at the previous time step.
        return paths