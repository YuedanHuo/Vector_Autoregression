class CSMC_nobs:
    def __init__(self, Num, phi, sigma0, y, z, B, j, fixed_particles, ESSmin=0.5):
        self.Num = Num  # Number of particles
        self.sigma0 = sigma0  # Variance for the prior of x_{j,0}
        self.phi = phi  # Variance for the random walk
        self.y = y  # Ay_{1:T}, T*N
        self.z = z  # z_{1:T}, T*(Np+1)
        self.B = B  # Matrix B for regression
        self.j = j  # Row index j
        self.fixed_particles = fixed_particles  # Fixed trajectory of a single particle to condition on
        self.ESSmin = ESSmin  # Minimum ESS threshold for resampling
        self.T = self.y.shape[0]

        # List to store particles history
        self.trajectories = np.zeros((Num, len(fixed_particles)))  # To store all particles' trajectories over time
        self.ancestors = np.zeros((Num, len(fixed_particles)), dtype=int)  # To store ancestor indices for each time step
        self.weights = np.zeros(self.Num)  # To store log weights over time
        #self.v_tilde = np.zeros(len(fixed_particles)) # estimate the volatility for update of Pi

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
        #self.v_tilde = self.v_tilde[indices] # update v_tilde
        self.ancestors[:, t] = indices  # Store ancestor indices for tracking

    def update_particles(self, t):
        # Update particles according to the dynamic model while fixing one trajectory
        self.particles[0] = self.fixed_particles[t]  # Fix the first particle's trajectory
        noise = np.random.normal(0, self.phi[self.j]**0.5, size=self.Num)
        self.particles[1:] += noise[1:]  # Update only non-fixed particles

        # in the case of not resampling, store the trajectory and ancestors
        self.trajectories[:, t] = self.particles  # Update trajectories
    

    def compute_ESS(self):
        # Compute the Effective Sample Size (ESS)
        weights_exp = np.exp(self.weights - np.max(self.weights))
        ESS = 1. / np.sum(weights_exp**2)
        return ESS

    def sampling_nobs(self):
        sampled_trajectory = np.zeros(self.T)
        sampled_ancestor = np.zeros(self.T, dtype=int)

        weights = np.exp(self.weights - np.max(self.weights))
        weights /= weights.sum()

        # Step 1: Sample the final state from the last time step
        sampled_index = np.searchsorted(np.cumsum(weights), np.random.rand(1))
        sampled_trajectory[-1] = self.trajectories[sampled_index, -1][0]
        sampled_ancestor[-1] = sampled_index

        for t in range(self.T-2, -1, -1):
            sample_index_new = self.ancestors[sampled_index, t]
            sampled_trajectory[t] = self.trajectories[sample_index_new,t]
            sampled_ancestor[t] = sample_index_new
            sampled_index = sample_index_new
        return sampled_trajectory, sampled_ancestor

    def run(self):
        v_tilde_mean = np.empty(self.T)
        # Step 1: Initialize particles
        self.initialize_particles()

        # Step 2: Iterate through time steps
        for t in range(self.T):
            self.ancestors[:, t] = np.arange(self.Num)
            #Compute ESS and resample if necessary (ESS < threshold)
            ESS = self.compute_ESS()
            if ESS < self.ESSmin * self.Num:
                self.resample(t-1)  # Trigger resampling if ESS is too low

            # Step 3: Propagate particles and fix one particle's trajectory
            self.update_particles(t)

            # Step 4: Compute weights based on new particle positions
            log_weights = self.compute_log_weights(self.particles, t)
            self.weights += log_weights  # Keep track of accumulated log weights
            self.weights -= np.max(self.weights)  # Normalize weights


        # Perform backward sampling to select one trajectory
        sampled_trajectory, sampled_ancestor = self.sampling_nobs()
        #v_tilde_mean = np.exp(sampled_trajectory)*np.random.normal(0, 1, size=self.T)
        v_tilde_mean = None

        return self.trajectories, self.ancestors, sampled_trajectory, sampled_ancestor, v_tilde_mean
    

Nums = [30]
smc = SMC(Num=1, phi = phi , sigma0=sigma_0, y=Ay, z=z_test, B =B, j = 0)
final_particles = smc.run()
n_iter = 2000
time_csmc_nobs = []
update_rate_nobs = np.zeros ((len(Nums), T))
for k,Num in enumerate(Nums):
  sampled_trajactoies = np.zeros((n_iter,T))
  sampled_ancestors = np.zeros((n_iter,T))
  csmc = CSMC_nobs(Num=Num, phi=phi, sigma0=sigma_0, y=Ay, z=z_test, B=B, j=0, fixed_particles= final_particles[:,0])
  start_time = time.time()
  for n in tqdm(range(n_iter)):
    trajectories, ancestors, sampled_trajectory, sampled_ancestor,_ = csmc.run()
    sampled_trajactoies[n,:] = sampled_trajectory
    sampled_ancestors[n,:] = sampled_ancestor
  end_time = time.time()
  time_csmc_nobs.append(end_time - start_time)
  for t in range(T):
    update_rate_nobs[k,t] = np.mean(1-np.isclose(sampled_trajactoies[:, t], final_particles[t, 0]))


plt.plot(update_rate_nobs[0,:], label='Without BS')
plt.plot(update_rate[5,:], label = 'With BS')
plt.xlabel('Time')
plt.ylabel('Update Rate')
plt.title('Update Rate of CSMC')
plt.legend()
plt.grid(True)