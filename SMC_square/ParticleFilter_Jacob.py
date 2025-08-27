import numpy as np
from scipy.special import logsumexp

import gc

class AncestryNode:
    def __init__(self, state, node_registry, parent_node=None):
        self.state = state
        self.parent = parent_node
        # for debugging
        node_registry.add(self)

    def get_path(self):
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        return np.array(path[::-1]).squeeze()



# callable functions for the PF class
# specify for VAR model
def sample_prior_fn(N, statedim, sigma_0 = 2.31):
    """Sample from initial state prior p(x_0 | theta)"""
    return np.random.normal(0, sigma_0, size=(N, statedim)) # Ensure (N, state_dim) shape


def transition_fn(x_prev, theta):
    """
    phi_j : the jth diagonal element of the current phi
    """
    # x_prev is (N_x, state_dim)
    phi_j = theta['Phi']
    return x_prev +  np.random.normal(0, phi_j**0.5, size= x_prev.shape)


def log_likelihood_fn(x, y_t, z_t, theta):
    """Compute log-likelihood p(y_t | x_t, theta)"""
    # x_t is the log volatility
    # y_t observation, z_t the lagged 
    A_j = theta['A']
    Aphi_j = theta['Api']
    res = (A_j @ y_t - Aphi_j @ z_t).squeeze()
    res = np.clip(res, -1e3, 1e3)

    lw = -0.5 * (np.log(2 * np.pi) + x + np.exp(-x) * res**2)
    lw = np.where(np.isnan(lw), -np.inf, lw)
    # In a 1D model, x_t[:, 0] accesses the state value
    return lw

# --- Particle Filter with Jacob's Method Philosophy (Core) ---
class ParticleFilterJacob:
    def __init__(self, N_x, T,theta = {}, state_dim=1):
        self.N_x = N_x
        self.T = T
        self.state_dim = state_dim # enable multidim PF 

        self.particles = np.zeros((N_x, state_dim))
        self.log_weights = np.zeros(N_x)

        self.ancestry_nodes = [None] * N_x 
        
        self.log_likelihood = 0.0
        self.threshold = 0.5

        self.sample_prior_fn = sample_prior_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.transition_fn = transition_fn
        self.theta = theta
        # theta contains A, Phi, Api in VAR 
        # if for each k in [K] we have a single object, then it should contain the kth row
    
    def reset_ancestry(self):
        self.ancestry_nodes = [None] * self.N_x

    
    def initialize(self, node_registry):
        self.sample_prior_fn = sample_prior_fn
        self.log_likelihood_fn = log_likelihood_fn
        #self.theta = theta 

        x_0 = self.sample_prior_fn(self.N_x, self.state_dim)
        
        self.particles = x_0

        for i in range(self.N_x):
            node = AncestryNode(x_0[i], node_registry, parent_node=None)
            self.ancestry_nodes[i] = node

    def step(self, t, y_t,z_t, node_registry):

        prev_ancestry_nodes = self.ancestry_nodes 

        log_w_prev = self.log_weights - logsumexp(self.log_weights)
        w_prev = np.exp(log_w_prev)
        w_prev /= np.sum(w_prev)
        
        ess = 1.0 / np.sum(w_prev ** 2)
        if ess < self.threshold * self.N_x:
            a = np.random.choice(self.N_x, size=self.N_x, p=np.squeeze(w_prev))
            # reset log weights
            self.log_weights = np.zeros(self.N_x)
        else:
            a = np.arange(self.N_x)

        x_prev_resampled = self.particles[a]
        x_t = self.transition_fn(x_prev_resampled, self.theta)
        # add clipping 
        x_t = np.clip(x_t, a_min=-10.0, a_max=10.0)

        lw_t = self.log_likelihood_fn(x_t, y_t,z_t, self.theta)

        self.particles = x_t
        self.log_weights += np.squeeze(lw_t) #- logsumexp(lw_t)
        self.log_likelihood = logsumexp(self.log_weights) - np.log(self.N_x)

        new_ancestry_nodes = [None] * self.N_x
        for i in range(self.N_x):
            parent_node = prev_ancestry_nodes[a[i]] 
            node = AncestryNode(x_t[i], node_registry, parent_node=parent_node)
            new_ancestry_nodes[i] = node
        
        self.ancestry_nodes = new_ancestry_nodes 

    def get_marginal_loglik(self):
        return self.log_likelihood

    def sample_trajectory(self):
        log_w = self.log_weights - logsumexp(self.log_weights)
        final_idx = np.random.choice(self.N_x, p=np.squeeze(np.exp(log_w)))
        final_node = self.ancestry_nodes[final_idx]
        return final_node.get_path()

    def load_from_matrix_form(self, trajectories_matrix, ancestor_indices_matrix, node_registry, final_log_weights=None):
        """
        Loads particle states and ancestry from a matrix-based representation
        into the AncestryNode linked-list storage.

        Args:
            trajectories_matrix (np.array): A NumPy array of shape (N_x, T_total, state_dim)
                                            containing particle states for all time steps.
            ancestor_indices_matrix (np.array): A NumPy array of shape (N_x, T_total)
                                                where ancestor_indices_matrix[i, t] is the
                                                index of the parent of particle i at time t
                                                (from time t-1). For t=0, this should be -1
                                                or a similar indicator for no parent.
            final_log_weights (np.array, optional): The normalized log-weights of the particles
                                                    at the *last* time step (T_total).
                                                    If None, these will be initialized to uniform
                                                    weights for path sampling purposes.
        
        Raises:
            ValueError: If dimensions do not match the instance's N_x, T, state_dim.
        """
        
        # Infer dimensions from the input matrices
        N_x_input, T_total_input, state_dim_input = trajectories_matrix.shape

        # Basic dimension checks
        if N_x_input != self.N_x:
            raise ValueError(f"Input N_x ({N_x_input}) does not match filter's N_x ({self.N_x}).")
        if state_dim_input != self.state_dim:
            raise ValueError(f"Input state_dim ({state_dim_input}) does not match filter's state_dim ({self.state_dim}).")
        
        # clear the ancestory for garbage collection
        self.reset_ancestry()
        #gc.collect()

        # Initialize a temporary list to hold the nodes for each time step
        # nodes_at_time[t] will be a list of AncestryNode objects for particles at time t
        all_generations_nodes = [None] * (T_total_input)

        # Process time step 0 (initial particles)
        current_gen_nodes = [None] * self.N_x
        for i in range(self.N_x):
            state = trajectories_matrix[i, 0, :]
            node = AncestryNode(state,node_registry, parent_node=None) # t=0 particles have no parent
            current_gen_nodes[i] = node
        all_generations_nodes[0] = current_gen_nodes

        # Process subsequent time steps from t=1 to T
        for t in range(1, T_total_input):
            prev_gen_nodes = all_generations_nodes[t-1]
            current_gen_nodes = [None] * self.N_x
            for i in range(self.N_x):
                state = trajectories_matrix[i, t, :]
                parent_idx = ancestor_indices_matrix[i, t]

                if parent_idx < 0 or parent_idx >= self.N_x:
                    raise ValueError(f"Invalid parent index {parent_idx} for particle {i} at time {t}. "
                                     "Parent indices must be between 0 and N_x-1 for t>0. For t=0, they should be invalid (e.g., -1).")
                
                parent_node = prev_gen_nodes[parent_idx]
                node = AncestryNode(state, node_registry,parent_node=parent_node)
                current_gen_nodes[i] = node
            all_generations_nodes[t] = current_gen_nodes

        # After building the entire tree, update the filter's current state to the last time step
        self.particles = trajectories_matrix[:, -1, :] # Set particles to the last generation's states
        self.ancestry_nodes = all_generations_nodes[-1] # Set ancestry_nodes to the last generation's nodes

        # Set final log-weights
        if final_log_weights is not None:
            if final_log_weights.shape != (self.N_x,):
                raise ValueError(f"final_log_weights shape {final_log_weights.shape} does not match expected ({self.N_x},).")
            self.log_weights = final_log_weights
        else:
            # currently we resample at the end of CSMC
            # so all particles trajectories have the same weight
            self.log_weights = np.zeros(self.N_x)

        # initialize log likelihood 
        self.log_likelihood = 0.0 
        #print(f"Particle filter loaded with {self.N_x} particles over {T_total_input_plus_1} time steps.")