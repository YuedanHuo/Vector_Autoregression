import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi
from Pi_update import update_Pi_corrected
from SMC import SMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/MH_gibbs')
from Lambda_update import sample_lambdas

class GibbsSampler_oldf:
    def __init__(self, y, z, mu_A, Sigma_A, Pi_prior_mean, Pi_prior_var, Pi_prior_var_inv, phi, sigma0, T, N, p, Num=30):
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
        self.Num = Num  # Number of step in MH in sampling of lambdas
        self.A = None  # A matrix
        self.Pi = None  # Pi matrix
        self.B = None  # B = A @ Pi
        self.Ay = None  # Ay matrix
        self.log_lambdas = np.empty((self.N, self.T))  # log-lambda values
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

    def update_Pi_corrected(self):
        self.Pi  = update_Pi_corrected(self.Ay, self.z, self.Pi, self.log_lambdas, self.p, self.A, self.Pi_prior_mean, self.Pi_prior_var_inv)

    def update_log_lambdas(self):
        # Update log-lambdas using CSMC
        Ay_new = (self.A @ self.y.T).T
        u_new = Ay_new - (self.z @ self.B.T)
        lambda_new = np.zeros((self.N, self.T))
        for j in range(self.N):
            lambda_new[j, :] = sample_lambdas(u_new[:, j], np.exp(self.log_lambdas[j, :]), self.phi, j, num_iter=self.Num)

        self.log_lambdas = np.log(lambda_new)

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

        for _ in tqdm.tqdm(range(num_iterations)):
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

