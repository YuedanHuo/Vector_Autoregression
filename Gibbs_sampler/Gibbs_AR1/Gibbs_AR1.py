import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/Gibbs_AR1')
from SMC_AR1 import SMC
from CSMC_AR1 import CSMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/APi_Gibbs')
from Update_APi import update_APi
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/Gibbs_AR1')
import importlib, update_mu
importlib.reload(update_mu)
from update_mu import update_mu_vectorized
from update_rho import update_rho


from scipy.linalg import solve_triangular
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import tqdm


class GibbsSampler_AR1:
    def __init__(self, y, z, mu_A, Sigma_A, Pi_prior_mean, Pi_prior_var, Pi_prior_var_inv, phi, sigma0, T, N, p, phi_1 = 20, phi_2 = 1.5, Num=30):
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
        self.Num = Num  # Number of particles for SMC/CSMC
        self.A = None  # A matrix
        self.Pi = None  # Pi matrix
        self.B = None  # B = A @ Pi
        self.log_lambdas = None  # log-lambda values
        self.ancestors = None
        self.trajectories = None # add this for SMC^2 initialization
        self.Ay = None
        self.volatilities = None  # Stores volatilities

        self.mu = None # the long run mean of the AR1 process of valatility
        self.rho = None # both of them should be vectors of length K 
        self.phi_1 = phi_1
        self.phi_2 = phi_2 # prior parameter for rho

    def initialize(self):
        # initialize the AR1 parameter
        self.mu = np.zeros(self.N) # for mu initialize it as 0
        self.rho = np.random.beta(self.phi_1, self.phi_2, self.N) # for rho draw from the prior 

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
            smc = SMC(Num=1, phi=self.phi, sigma0=self.sigma0, y=self.Ay, z=self.z, B=self.B, j=j, mu = self.mu[j], rho = self.rho[j])
            final_particles = smc.run()
            self.log_lambdas[j, :] = final_particles.reshape(-1)


    def update_A(self):
        # Update A using its conditional posterior
        self.A = compute_posterior_A_with_log_lambdas(
            self.y, self.z, self.log_lambdas, self.mu_A, self.Sigma_A, self.Pi
        )

    def update_Pi(self, first_iter):
        self.Pi  = update_APi(self.Ay, self.z, self.Pi, self.log_lambdas, self.p, self.A, self.Pi_prior_mean, self.Pi_prior_var_inv, first_iter)

    def update_log_lambdas(self):
        # Update log-lambdas using CSMC
        log_lambda_new = np.zeros((self.N, self.T))
        ancestors_new = np.zeros((self.Num,self.N, self.T))
        trejactories_new = np.zeros((self.Num,self.N, self.T))
        for j in range(self.N):
            csmc = CSMC(
                Num=self.Num,
                phi=self.phi,
                sigma0=self.sigma0,
                y= self.Ay,
                z=self.z,
                B=self.B,
                j=j,
                fixed_particles=self.log_lambdas[j, :],
                mu = self.mu[j],
                rho = self.rho[j],
            )
            trejactories_new[:,j,:], ancestors_new[:,j,:], log_lambda_new[j, :], _, _ = csmc.run()
        self.log_lambdas = log_lambda_new
        self.ancestors = ancestors_new
        self.trejactories = trejactories_new

    def update_phi(self):
        self.phi = compute_posterior_phi(self.log_lambdas, df_phi = self.N+2)

    def update_ar1_param(self):
        for j in range(self.N):
            new_rho_j = update_rho(self.log_lambdas[j,:], self.phi[j], self.mu[j], self.phi_1, self.phi_2, self.rho[j])
            self.rho[j] = new_rho_j

        self.mu = update_mu_vectorized(self.log_lambdas, self.rho, self.phi)

    def run(self, num_iterations):
        # Run the Gibbs sampler for a specified number of iteration
        self.initialize()
        self.samples = {
            "A": [],
            "Pi": [],
            "log_lambdas": [],
            'phi':[],
            'ancestors': [],
            'trejactories': [],
            'rho' : [],
            'mu': [],
        }
        
        first_iter = True # bool indicator to skip the rejection sampling for Pi in the fist iteration
        # when params are from prior
        for _ in tqdm.tqdm(range(num_iterations)):
            self.update_phi()
            self.update_Pi(first_iter)
            self.B = self.A @ self.Pi  # Update B after each iteration
            self.update_A()
            self.Ay = (self.A @ self.y.T).T
            self.B = self.A @ self.Pi
            #first_iter = False # skip this for now
            self.update_log_lambdas()
            self.update_ar1_param()


            # Store samples
            self.samples["A"].append(self.A.copy())
            self.samples["Pi"].append(self.Pi.copy())
            self.samples["log_lambdas"].append(self.log_lambdas.copy())
            self.samples["phi"].append(self.phi.copy())
            self.samples['ancestors'].append(self.ancestors.copy())
            self.samples['trejactories'].append(self.trejactories.copy())
            self.samples['rho'].append(self.rho.copy())
            self.samples['mu'].append(self.mu.copy())
        return self.samples