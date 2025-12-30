import numpy as np
from numba import njit

# --- Numba helper functions ---

@njit(nogil=True)
def compute_log_weights_numba(particles, B_row, z_row, y_jt):
    Num = particles.shape[0]
    logw = np.empty(Num)
    residual = y_jt - np.dot(B_row, z_row)
    residual = min(max(residual, -1e3), 1e3)
    for i in range(Num):
        logw[i] = -0.5 * (particles[i] + np.exp(-particles[i]) * (residual**2))
    return logw

@njit(nogil=True)
def update_particles_numba(particles, phi_j, fixed_particle):
    Num = particles.shape[0]
    noise = np.random.normal(0.0, np.sqrt(phi_j), Num)
    for i in range(1, Num):
        particles[i] += noise[i]
    particles[0] = fixed_particle
    for i in range(Num):
        if particles[i] > 10.0:
            particles[i] = 10.0
        elif particles[i] < -10.0:
            particles[i] = -10.0
    return particles

@njit(nogil=True)
def compute_ESS_numba(log_weights):
    max_logw = np.max(log_weights)
    weights = np.exp(log_weights - max_logw)
    weights /= np.sum(weights)
    ESS = 1.0 / np.sum(weights**2)
    return ESS

@njit(nogil=True)
def backward_sampling_numba(trajectories, all_log_weights, phi_j):
    Num, T = trajectories.shape
    sampled_traj = np.empty(T)
    sampled_ancestor = np.empty(T, dtype=np.int32)

    logw = all_log_weights[:, T-1]
    max_logw = np.max(logw)
    weights = np.exp(logw - max_logw)
    weights /= np.sum(weights)
    # multinomial sampling
    r = np.random.rand()
    cumw = 0.0
    for i in range(Num):
        cumw += weights[i]
        if r <= cumw:
            sampled_index = i
            break
    sampled_traj[T-1] = trajectories[sampled_index, T-1]
    sampled_ancestor[T-1] = sampled_index

    for t in range(T-2, -1, -1):
        particles_t = trajectories[:, t]
        logw = all_log_weights[:, t]
        particle_next = sampled_traj[t+1]
        logw += -(particle_next - particles_t)**2 / (2.0 * phi_j)
        max_logw = np.max(logw)
        weights = np.exp(logw - max_logw)
        weights /= np.sum(weights)
        r = np.random.rand()
        cumw = 0.0
        for i in range(Num):
            cumw += weights[i]
            if r <= cumw:
                sampled_index = i
                break
        sampled_traj[t] = particles_t[sampled_index]
        sampled_ancestor[t] = sampled_index

    return sampled_traj, sampled_ancestor

# --- CSMC Class ---

class CSMC:
    def __init__(self, Num, phi, sigma0, y, z, B, j, fixed_particles, ESSmin=0.5):
        self.Num = Num
        self.sigma0 = sigma0
        self.phi = phi
        self.y = y
        self.z = z
        self.B = B
        self.j = j
        self.fixed_particles = fixed_particles
        self.ESSmin = ESSmin
        self.T = len(fixed_particles)

        self.trajectories = np.zeros((Num, self.T))
        self.all_weights = np.zeros((Num, self.T))
        self.ancestors = np.zeros((Num, self.T), dtype=int)
        self.weights = np.zeros(self.Num)
        self.particles = np.zeros(self.Num)

    def initialize_particles(self):
        self.particles = np.random.normal(0.0, np.sqrt(self.sigma0), self.Num)
        self.weights = compute_log_weights_numba(self.particles, self.B[self.j,:], self.z[0,:], self.y[0,self.j])

    def compute_ESS(self):
        return compute_ESS_numba(self.weights)

    def resample(self, t):
        max_logw = np.max(self.weights)
        weights = np.exp(self.weights - max_logw)
        weights /= np.sum(weights)
        indices = np.searchsorted(np.cumsum(weights), np.random.rand(self.Num))
        indices[0] = 0
        self.particles = self.particles[indices]
        self.ancestors[:, t] = indices
        self.weights[:] = 0.0
        return indices

    def update_particles(self, t):
        self.particles = update_particles_numba(self.particles, self.phi[self.j], self.fixed_particles[t])
        self.trajectories[:, t] = self.particles

    def run(self, if_reconstruct=False):
        self.initialize_particles()
        for t in range(self.T):
            self.ancestors[:, t] = np.arange(self.Num)
            if self.compute_ESS() < self.ESSmin * self.Num:
                self.resample(t)
            self.update_particles(t)
            logw = compute_log_weights_numba(self.particles, self.B[self.j,:], self.z[t,:], self.y[t,self.j])
            self.all_weights[:, t] = logw
            self.weights += logw

        if if_reconstruct:
            self.resample(self.T - 1)

        sampled_trajectory, sampled_ancestor = backward_sampling_numba(self.trajectories, self.all_weights, self.phi[self.j])
        return self.trajectories, self.ancestors, sampled_trajectory, sampled_ancestor, self.weights


@njit(nogil=True)
def run_csmc_full_numba(Num, phi_val, sigma0, y_col, z, B_row, fixed_particles, ESSmin):
    """
    This function replaces the entire CSMC class instance and its .run() method.
    It takes raw arrays/scalars and returns raw arrays.
    """
    T = len(fixed_particles)
    
    # 1. Memory Allocation (Done inside Numba - very fast, no GIL)
    trajectories = np.zeros((Num, T),dtype=np.float32)
    all_weights = np.zeros((Num, T))
    ancestors = np.zeros((Num, T), dtype=np.int32)
    particles = np.zeros(Num)
    
    # 2. Initialization
    particles = np.random.normal(0.0, np.sqrt(sigma0), Num)
    
    # Inline compute_log_weights to save function overhead
    weights_log = compute_log_weights_numba(particles, B_row, z[0, :], y_col[0])
    
    # Accumulator for total weights (replacing self.weights)
    current_log_weights = weights_log.copy()

    # 3. The Main Loop (Entirely in C-speed, NO GIL)
    for t in range(T):
        ancestors[:, t] = np.arange(Num)
        
        # --- ESS Check & Resample ---
        # We inline the ESS logic to avoid overhead
        max_logw = np.max(current_log_weights)
        w_temp = np.exp(current_log_weights - max_logw)
        w_sum = np.sum(w_temp)
        w_norm = w_temp / w_sum
        ESS = 1.0 / np.sum(w_norm**2)
        
        if ESS < ESSmin * Num:
            # Resampling
            # Numba supports np.searchsorted and cumsum
            cdf = np.cumsum(w_norm)
            cdf[-1] = 1.0 # Ensure last is exactly 1 to avoid float errors
            u = np.random.rand(Num)
            # Stratified/Systematic resampling is usually better, but random is fine:
            indices = np.searchsorted(cdf, u)
            
            # Correction: Explicitly ensure 0 is 0 
            indices[0] = 0 
            
            # Reorder particles
            particles = particles[indices]
            ancestors[:, t] = indices
            current_log_weights[:] = 0.0 # Reset weights after resample
            
        # --- Update Particles ---
        particles = update_particles_numba(particles, phi_val, fixed_particles[t])
        trajectories[:, t] = particles
        
        # --- Weighting ---
        logw_step = compute_log_weights_numba(particles, B_row, z[t, :], y_col[t])
        
        all_weights[:, t] = logw_step
        current_log_weights += logw_step

    # 4. Backward Sampling
    sampled_traj, sampled_ancestor = backward_sampling_numba(trajectories, all_weights, phi_val)
    
    # Return everything needed
    return trajectories, ancestors, sampled_traj, sampled_ancestor, current_log_weights
