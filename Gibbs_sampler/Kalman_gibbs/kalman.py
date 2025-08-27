# initialize the mixture gaussian distribution
K = 7
m_list = np.array([-10.12999,-3.97281,-8.56686,2.77786,0.61942,1.79518,-1.08819]) - 1.2704
sigma_sqr_list = np.array([5.79596,2.61369,5.17950, 0.16735, 0.64009,0.34023,1.26261])
q_list = np.array([0.00730,0.10556,0.00002,0.04395,0.34001,0.24566,0.25750])

def generate_s (v,x_current,q_list,m_list,sigma_sqr_list):
  N,T = v.shape
  K = len(q_list)
  s = np.empty((N,T), dtype=int)
  for t in range(T):
    for n in range(N):
      prob = np.zeros(K)
      for k in range(K):
        prob[k] = q_list[k]*np.exp(-(v[n,t]-(m_list[k]-x_current[n,t]))**2/(2*sigma_sqr_list[k]))
        cdf = np.cumsum(prob/np.sum(prob))
        s[n,t] = np.searchsorted(cdf, np.random.rand(1))
  return s


def Kalman_forward_filtering(s, phi, sigma_sqr_list, m_list, v, sigma_0):
    '''
    Kalman filter for a multivariate model with Gaussian mixture emissions.

    Parameters:
    - s: (N, T) matrix, indicates which Gaussian mixture component is used.
    - phi: (N, N) state transition matrix.
    - sigma_sqr_list: (K,) list of variances for the K Gaussian mixtures.
    - m_list: (K,) list of means for the K Gaussian mixtures.
    - v: (N, T) matrix, observations, log(A@v_t**2+c)
    '''
    N, T = v.shape
    K = len(sigma_sqr_list)

    x_filter = np.empty((N, T))
    cov_filter = np.empty((N, N, T))

    # Initialization
    x = np.zeros(N)  # Initial state estimate (x_1|0)
    P = np.eye(N) * sigma_0  # Initial state covariance (P_1|0)

    for t in range(T):
        # **Prediction Step**
        sigma = np.array([sigma_sqr_list[s[i, t]] for i in range(N)])  # Observation variance based on Gaussian component
        m = np.array([m_list[s[i, t]] for i in range(N)])  # Means based on Gaussian component

        # Predicted state and covariance
        x_pred = x  # x_{t|t-1} = x_{t-1|t-1}
        P_pred = P + phi  # P_{t|t-1} = P_{t-1|t-1} + Phi

        # **Update Step**
        # Kalman gain: K_t = P_{t|t-1} * (P_{t|t-1} + diag(sigma))^{-1}
        K = P_pred @ np.linalg.inv(P_pred + np.diag(sigma))

        # Residual (innovation): mu_t = v_t^* - (m_{s_t} - 1.2704) - x_{t|t-1}
        mu = v[:, t] - m - x_pred

        # Updated state estimate: x_{t|t} = x_{t|t-1} + K_t * mu_t
        x = x_pred + K @ mu

        # Updated state covariance: P_{t|t} = (I - K_t) * P_{t|t-1}
        P = (np.eye(N) - K) @ P_pred

        # Store filtered state and covariance
        x_filter[:, t] = x
        cov_filter[:, :, t] = P

        # No additional update to P here, it should only be updated after prediction step and once in the update step

    return x_filter, cov_filter

def Kalman_backward_sampling(s, phi, sigma_sqr_list, m_list, v, x_filter, P_filter):
  N, T = v.shape
  L = np.linalg.cholesky(phi)
  L_inv = np.linalg.solve(L,np.eye(N))
  x_draw = np.empty((N,T))
  x_draw[:,T-1] = np.random.multivariate_normal(x_filter[:,T-1], P_filter[:,:,T-1])
  for t in range(T-2, -1, -1):
    #initialize
    x = x_filter[:,t]
    P = P_filter[:,:,t]
    tilde_x = L_inv@x_draw[:,t+1]
    for n in range(N):
      epsilon = tilde_x[n] - L_inv[n,:].T@x
      R = L_inv[n,:]@P@L_inv[n,:].T + 1
      x =  x+ (P@L_inv[n,:].T)*(epsilon/R)
      P = P - P@np.outer(L_inv[n,:],L_inv[n,:])@P/R
    x_draw[:,t] = np.random.multivariate_normal(x, P)
  return x_draw

class GibbsSampler_kalman_filter:
    def __init__(self, y, z, mu_A, Sigma_A, Pi_prior_mean, Pi_prior_var, Pi_prior_var_inv, phi, sigma0, T, N, p):
        self.y = y  # Observations
        self.z = z  # Predictors
        self.mu_A = mu_A  # Prior mean for A
        self.Sigma_A = Sigma_A  # Prior covariance for A
        self.Pi_prior_mean = Pi_prior_mean  # Prior mean for Pi
        self.Pi_prior_var = Pi_prior_var  # Prior covariance for Pi
        self.Pi_prior_var_inv = Pi_prior_var_inv  # Inverse of prior covariance for Pi
        self.phi = phi  # Variance for random walk of log-lambdas
        self.sigma0 = sigma0  # Prior variance for initial state of log-lambdas
        self.T = T  # Number of time steps
        self.N = N  # Number of variables
        self.p = p  # Number of lags
        self.A = None  # A matrix
        self.Pi = None  # Pi matrix
        self.B = None  # B = A @ Pi
        self.Ay = None  # Ay matrix
        self.log_lambdas = np.empty((self.N, self.T))  # log-lambda values
        self.m_list = np.array([-10.12999,-3.97281,-8.56686,2.77786,0.61942,1.79518,-1.08819]) - 1.2704
        self.sigma_sqr_list = np.array([5.79596,2.61369,5.17950, 0.16735, 0.64009,0.34023,1.26261])
        self.q_list = np.array([0.00730,0.10556,0.00002,0.04395,0.34001,0.24566,0.25750])
        #self.volatilities = np.empty((self.T,self.N,self.N))  # Stores volatilities

    def initialize(self):
        # Initialize A
        self.A = np.eye(self.N)
        for i in range(1, self.N):
            self.A[i, :i] = np.random.multivariate_normal(self.mu_A[:i], self.Sigma_A[:i, :i])

        #try initalize Pi by fitting a non-bayesian model
        #so as to stablize residuals
        model = VAR(self.y)
        results = model.fit(self.p)
        # Extract lag coefficients and reshape
        lag_coeffs = results.coefs.transpose(1, 0, 2).reshape(self.N, -1)  # Shape (N, N*lags)

        # Extract intercept
        intercept = results.intercept.reshape(self.N, 1)  # Shape (N, 1)

        # Combine intercept and lag coefficients
        Pi_init = np.hstack([intercept, lag_coeffs])  # Final shape (N, N*lags + 1)

        self.Pi = Pi_init
        self.B = self.A @ self.Pi
        self.Ay = (self.A @ self.y.T).T


        # Initialize log-lambdas
        self.log_lambdas = np.zeros((self.N, self.T))
        self.Ay = (self.A @ self.y.T).T
        for j in range(self.N):
            smc = SMC(Num=1, phi=self.phi, sigma0=self.sigma0, y=self.Ay, z=self.z, B=self.B, j=j)
            final_particles = smc.run()
            self.log_lambdas[j, :] = final_particles.reshape(-1)

    def update_A(self):
        # Update A using its conditional posterior
        self.A = compute_posterior_A_with_log_lambdas(
            self.y, self.z, self.log_lambdas, self.mu_A, self.Sigma_A, self.Pi
        )

    def update_Pi(self):
        # Update Pi using its conditional posterior

        #self.volatilities = np.empty((self.T, self.N, self.N))
        inv_A = solve_triangular(self.A, np.eye(self.A.shape[0]), lower=True)
        for t in range(self.T):
            Lambda_t_sr = np.diag(np.exp(0.5 * self.log_lambdas[:, t]))
            self.volatilities[t, :, :] = inv_A @ Lambda_t_sr

        self.Pi = update_Pi_gibbs(
            self.y, self.z, self.Pi, self.volatilities, self.p, self.Pi_prior_mean, self.Pi_prior_var, self.Pi_prior_var_inv
        )

    def update_Pi_corrected(self):
        self.Pi  = update_Pi_corrected(self.Ay, self.z, self.Pi, self.log_lambdas, self.p, self.A, self.Pi_prior_mean, self.Pi_prior_var_inv)

    def update_log_lambdas(self):
        # Update log-lambdas using CSMC
        Ay_new = (self.A @ self.y.T).T
        u_new = Ay_new - (self.z @ self.B.T)
        v_new = np.log(u_new**2+0.001).T
        lambda_new = np.zeros((self.N, self.T))
        s = generate_s(v_new, self.log_lambdas, self.q_list, self.m_list, self.sigma_sqr_list)
        # so far we use phi diagonal, so phi is a N array
        mean_filter, cov_filter = Kalman_forward_filtering(s, np.diag(self.phi), self.sigma_sqr_list, self.m_list, v_new, self.sigma0)
        self.log_lambdas = Kalman_backward_sampling(s, np.diag(self.phi), self.sigma_sqr_list, self.m_list, v_new, mean_filter, cov_filter)

    def updata_phi(self):
      self.phi = compute_posterior_phi(self.log_lambdas, df_phi = self.N+2 )

    def run(self, num_iterations):
        # Run the Gibbs sampler for a specified number of iterations
        self.initialize()
        self.samples = {
            "A": [],
            "Pi": [],
            "log_lambdas": [],
            'phi': []
        }

        for _ in tqdm(range(num_iterations)):
            #cProfile.run('self.update_A()')  # Profiling the A update
            #cProfile.run('self.update_Pi()')  # Profiling the Pi update
            #cProfile.run('self.update_log_lambdas()')
            self.update_A()
            self.Ay = (self.A @ self.y.T).T
            self.B = self.A @ self.Pi
            self.update_log_lambdas()
            self.update_Pi_corrected()
            self.B = self.A @ self.Pi  # Update B after each iteration
            self.updata_phi()

            # Store samples
            self.samples["A"].append(self.A.copy())
            self.samples["Pi"].append(self.Pi.copy())
            self.samples["log_lambdas"].append(self.log_lambdas.copy())
            self.samples["phi"].append(self.phi.copy())

        return self.samples