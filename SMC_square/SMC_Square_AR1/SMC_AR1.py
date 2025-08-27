import numpy as np

class SMC:
    """
    Implements a Sequential Monte Carlo (SMC) filter, also known as a particle filter.
    This class is designed to estimate the state of a dynamic system over time,
    given a sequence of observations.
    """
    def __init__(self, Num, phi, sigma0, y, z, B, j, mu = 0, rho = 1):
        """
        Initializes the SMC filter with model parameters and data.

        Args:
            Num (int): The number of particles to use in the filter.
            phi (float): The variance parameter for the state transition (random walk).
            sigma0 (float): The variance for the prior distribution of the initial state x_{j,0}.
            y (np.ndarray): The observations array, with shape (T, N), where T is the number of time steps.
            z (np.ndarray): An auxiliary variable array, with shape (T, Np+1).
            B (np.ndarray): A matrix used in the observation model (B = A @ Pi).
            j (int): The specific row index in the observation and parameter arrays to be used.
        """
        # --- Model and Data Parameters ---
        self.Num = Num          # Number of particles
        self.sigma0 = sigma0    # Variance for the prior of x_{j,0}
        self.phi = phi          # Variance for the random walk process noise
        self.y = y              # Observations, shape (T, N)
        self.z = z              # Auxiliary data, shape (T, Np+1)
        self.B = B              # Matrix used in the observation model
        self.j = j              # Row index for the current process
        self.T = self.y.shape[0] # Total number of time steps

        # add AR1 parameters
        self.mu = mu
        self.rho = rho

        # --- Internal State ---
        self.particles = np.zeros(self.Num) # Array to hold the current particles
        self.weights = np.zeros(self.Num)   # Array to hold the current particle weights
        
        # --- History Storage ---
        self.particles_history = [] # List to store the particle sets at each time step

    def initialize_particles(self):
        """
        Initializes the particles at time t=0 by drawing from the prior distribution.
        The prior is assumed to be a normal distribution with mean 0.
        """
        # Draw initial particles from a Gaussian distribution N(0, sigma0)
        self.particles = np.random.normal(0., self.sigma0**0.5, size=self.Num)

    def compute_log_weights(self, particles, t):
        """
        Computes the logarithm of the unnormalized weights for each particle at a given time step t.
        The weight is based on the likelihood of the observation y[t] given the particle's state.

        Args:
            particles (np.ndarray): The current set of particles.
            t (int): The current time step.

        Returns:
            np.ndarray: An array containing the log-weight for each particle.
        """
        log_weights = np.zeros(self.Num)
        for i in range(self.Num):
            # Calculate the residual: the difference between the observation and the prediction
            # The clip is for numerical stability, preventing very large values.
            res = np.clip(self.y[t, self.j] - np.dot(self.B[self.j, :], self.z[t, :]), -1e3, 1e3)
            # This is the log-likelihood function specific to the model.
            # It appears to be a non-standard form, where the state `particles[i]` influences the precision.
            log_weights[i] = -0.5 * (particles[i] + np.exp(-particles[i]) * res**2)
        return log_weights

    def resample(self, t):
        """
        Resamples the particles based on their computed weights.
        This step mitigates particle degeneracy by eliminating low-weight particles
        and duplicating high-weight particles. It uses multinomial resampling.
        
        Args:
            t (int): The current time step to compute weights for.
        """
        # Compute the log-weights for the current set of particles at time t
        log_weights = self.compute_log_weights(self.particles, t)
        
        # Convert log-weights to normalized weights, using a trick for numerical stability
        # Subtracting the max log-weight before exponentiating prevents overflow/underflow.
        self.weights = np.exp(log_weights - np.max(log_weights))
        self.weights /= self.weights.sum()  # Normalize weights to sum to 1

        # Perform systematic resampling using inverse CDF method.
        indices = np.searchsorted(np.cumsum(self.weights), np.random.rand(self.Num))
        self.particles = self.particles[indices]

    def update_particles(self):
        """
        Propagates the particles to the next time step according to the system's dynamic model.
        In this case, the model is a simple random walk.
        """
        # Evolve each particle by adding Gaussian noise (random walk)
        noise = np.random.normal(0, self.phi[self.j]**0.5, size=self.Num)

        # add the AR1 process here
        self.particles = self.rho * (self.particles - self.mu) + self.mu + noise

        # Clip the particles to keep them within a plausible range, preventing divergence.
        self.particles = np.clip(self.particles, -10, 10)

    def run(self):
        """
        Executes the main loop of the SMC filter over all time steps.

        Returns:
            np.ndarray: A 2D array of shape (T, Num) containing the history of particles at each step.
        """
        # 1. Initialize particles at t=0
        self.initialize_particles()

        # 2. Iterate through each time step from t=0 to T-1
        for t in range(self.T):
            # The standard SMC loop is: Weight -> Resample -> Propagate.
            # The logic here is slightly different: for t>0, it re-weights and resamples
            # before the propagation step.
            if t > 0:
                self.resample(t) # Resample based on weights at time t
            
            # Propagate particles to the next state (for the next iteration)
            self.update_particles()
            
            # Store the current state of the particles
            self.particles_history.append(self.particles.copy())

        # Return the full history of particles as a numpy array
        return np.array(self.particles_history)