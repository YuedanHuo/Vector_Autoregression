
class CSMC:
    def __init__(self, Num, phi, sigma0, y, z, B, j, fixed_particles, ESSmin=0.5):
        self.Num = Num  # Number of particles
        self.sigma0 = sigma0  # Variance for the prior of x_{j,0}
        self.phi = phi  # Variance for the random walk
        self.y = y  # Ay_{1:T}, T*N
        self.z = z  # z_{1:T}, T*(Np+1)
        self.B = B  # Matrix B = A@Pi
        self.j = j  # Row index j
        self.fixed_particles = fixed_particles  # Fixed trajectory of a single particle to condition on
        self.ESSmin = ESSmin  # Minimum ESS threshold for resampling
        self.T = self.y.shape[0]

        # List to store particles history
        self.trajectories = np.zeros((Num, len(fixed_particles)))  # To store all particles' trajectories over time
        self.ancestors = np.zeros((Num, len(fixed_particles)), dtype=int)  # To store ancestor indices for each time step
        self.weights = np.zeros(self.Num)  # To store log weights over time
        
    def initialize_particles(self):
        # Initialize particles for the first time step
        self.particles = np.random.normal(0., self.sigma0**0.5, size=self.Num)
        self.weights = self.compute_log_weights(self.particles, 0)  # Initial weights

    def compute_log_weights(self, particles, t):
        # Compute log weights in a vectorized manner
        residuals = self.y[t, self.j] - np.dot(self.B[self.j, :], self.z[t, :])
        log_weights = -0.5 * (particles + np.exp(-particles) * (residuals**2))
        return log_weights

    def resample(self, t):
        # Normalize log weights by exponentiating and computing the normalized weights
        max_log_weight = np.max(self.weights)
        weights = np.exp(self.weights - max_log_weight)  # Compute normalized weights from log weights
        weights /= weights.sum()  # Normalize weights to sum to 1

        # Perform inverse CDF for multinomial resampling
        indices = np.searchsorted(np.cumsum(weights), np.random.rand(self.Num))

        self.weights = np.zeros(self.Num)  # Update log weights for the next iteration

        # Resample particles and store ancestors
        self.particles = self.particles[indices]
        self.trajectories[:, t] = self.particles  # Update trajectories
        self.ancestors[:, t] = indices  # Store ancestor indices for tracking

    def update_particles(self, t):
        # Update particles according to the dynamic model while fixing one trajectory
        self.particles[0] = self.fixed_particles[t]  # Fix the first particle's trajectory
        noise = np.random.normal(0, self.phi[self.j]**0.5, size=self.Num)
        self.particles[1:] += noise[1:]  # Update only non-fixed particles

        # in the case of not resampling, store the trajectory and ancestors
        self.trajectories[:, t] = self.particles  # Update trajectories
        
    def backward_sampling(self):
        # Backward sampling to select one trajectory
        sampled_trajectory = np.zeros(self.T)
        sampled_ancestor = np.zeros(self.T, dtype=int)

        # Initialize with the last particle's trajectory
        weights = self.compute_log_weights(self.particles, self.T-1) - np.max(self.compute_log_weights(self.particles, self.T-1))
        weights = np.exp(weights)
        weights /= weights.sum()  # Normalize weights


        # Step 1: Sample the final state from the last time step
        sampled_index = np.searchsorted(np.cumsum(weights), np.random.rand(1))
        #sampled_index = np.random.choice(range(self.Num), p=weights)
        sampled_trajectory[-1] = self.trajectories[sampled_index, -1][0]
        sampled_ancestor[-1] = sampled_index

        # Step 2: Backward sampling from time T-1 to 0
        for t in range(self.T-2, -1, -1):
            particles = self.trajectories[:, t]
            log_weights = self.compute_log_weights(particles, t) - np.max(self.compute_log_weights(particles, t))

            particle_next = sampled_trajectory[t+1] # the sampled trajectory that has already been chosen
            log_weights_tilde = log_weights -(particle_next - particles)**2 / (2 * self.phi[self.j]) # the adjusted weight
            weights_tilde = np.exp(log_weights_tilde - np.max(log_weights_tilde))
            weights_tilde /= weights_tilde.sum()  # Normalize weights_tilde

            sampled_index = np.searchsorted(np.cumsum(weights_tilde), np.random.rand(1))
            #sampled_index = np.random.choice(range(self.Num), p=weights_tilde)
            sampled_trajectory[t] = particles[sampled_index]
            sampled_ancestor[t] = sampled_index
        return sampled_trajectory, sampled_ancestor

    def compute_ESS(self):
        # Compute the Effective Sample Size (ESS)
        weights_exp = np.exp(self.weights - np.max(self.weights))
        ESS = 1. / np.sum(weights_exp**2)
        return ESS

    def run(self):
        v_tilde_mean = np.empty(self.T)
        # Step 1: Initialize particles
        self.initialize_particles()

        # Step 2: Iterate through time steps
        for t in range(self.T):
            self.ancestors[:, t] = np.arange(self.Num)  # Update ancestors
            #Compute ESS and resample if necessary (ESS < threshold)
            ESS = self.compute_ESS()
            if ESS < self.ESSmin * self.Num:
                self.resample(t)  # Trigger resampling if ESS is too low

            # Step 3: Propagate particles and fix one particle's trajectory
            self.update_particles(t)

            # Step 4: Compute weights based on new particle positions
            log_weights = self.compute_log_weights(self.particles, t)
            self.weights += log_weights  # Keep track of accumulated log weights
            self.weights -= np.max(self.weights)  # Normalize weights


        # Perform backward sampling to select one trajectory
        sampled_trajectory, sampled_ancestor = self.backward_sampling()
        v_tilde_mean = None

        return self.trajectories, self.ancestors, sampled_trajectory, sampled_ancestor, v_tilde_mean
