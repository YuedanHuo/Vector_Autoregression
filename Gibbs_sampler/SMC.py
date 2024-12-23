class SMC:
    def __init__(self, Num, phi, sigma0, y, z, B, j):
        self.Num = Num  # Number of particles
        self.sigma0 = sigma0  # Variance for the prior of x_{j,0}
        self.phi = phi  # Variance for the random walk
        self.y = y  # Ay_{1:T}, T*N
        self.z = z  # z_{1:T}, T*(Np+1)
        self.B = B  # B = A@Pi
        self.j = j  # Row index j
        self.T = self.y.shape[0]

        # List to store particles history
        self.particles_history = []

    def initialize_particles(self):
        # Initialize particles for the first time step
        self.particles = np.random.normal(0., self.sigma0**0.5, size=self.Num)
        #self.weights = self.compute_log_weights(self.particles, 0)  # Initial weights
        #self.resample(0)
        #self.particles_history.append(self.particles.copy())  # Store initial particles

    def compute_log_weights(self, particles, t):
        # Compute weights based on likelihood and prior
        log_weights = np.zeros(self.Num)
        for i in range(self.Num):
            log_weights[i] = -.5*(particles[i] + np.exp(-particles[i]) * (self.y[t, self.j] - np.dot(self.B[self.j, :], self.z[t, :]))**2)
        return log_weights

    def resample(self, t):
        # Resample particles based on the weights
        log_weights = self.compute_log_weights(self.particles, t)
        self.weights = np.exp(log_weights - np.max(log_weights))  # Stabilize weights
        self.weights /= self.weights.sum()  # Normalize weights to sum to 1
        #indices = np.random.choice(range(self.Num), size=self.Num, p=self.weights)
        # perform inverse CDF for multinomial resampling
        indices = np.searchsorted(np.cumsum(self.weights), np.random.rand(self.Num))
        self.particles = self.particles[indices]

    def update_particles(self):
        # Propagate particles based on the dynamic model (e.g., random walk)
        self.particles += np.random.normal(0, self.phi[self.j]**0.5, size=self.Num)

    def run(self):
        # Initialize particles at t=0
        self.initialize_particles()

        # Iterate through time steps t=1 to T
        for t in range(self.T):
            if t>0:
                self.weights = self.compute_log_weights(self.particles, t)
                self.resample(t-1)
            self.update_particles()
            self.particles_history.append(self.particles.copy())  # Store particles after each step

        return np.array(self.particles_history)  # Return history of particles
