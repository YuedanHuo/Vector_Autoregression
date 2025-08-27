import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
import sys
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler')
from A_update import compute_posterior_A_with_log_lambdas
from update_phi import compute_posterior_phi, compute_posterior_phi_by_component
from SMC import SMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/CSMC_gibbs')
from CSMC_bs import CSMC
sys.path.append('/Users/hildahuo/Desktop/course registraition/research project/VAR/var python code/Gibbs_sampler/APi_Gibbs')
from Update_APi import update_APi

from joblib import Parallel, delayed, parallel_backend

from scipy.linalg import solve_triangular
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import tqdm


class GibbsSampler_SMC_square:
    def __init__(self, y, z, k, mu_A, Sigma_A, Pi_prior_mean, Pi_prior_var, Pi_prior_var_inv, phi, sigma0, T, N, p, Num=30):
        self.y = y  # Observations
        self.z = z  # Predictors
        self.mu_A = mu_A  # Prior mean for A
        self.Sigma_A = Sigma_A  # Prior covariance for A
        self.Pi_prior_mean = Pi_prior_mean  # Prior mean for Pi
        self.Pi_prior_var = Pi_prior_var  # Prior covariance for Pi
        self.Pi_prior_var_inv = Pi_prior_var_inv  # Inverse of prior covariance for Pi
        self.Phi = phi  # Variance for random walk of log-lambdas
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

        self.k = k ## we only observe up to k at the last time point T

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
        ######## change it here
        self.log_lambdas = np.full((self.N, self.T), np.inf)
        self.Ay = (self.A @ self.y.T).T
        for j in range(self.N):
            if j < self.k: ### self.k the part that we see at t
                smc = SMC(Num=1, phi=self.Phi, sigma0=self.sigma0, y=self.Ay, z=self.z, B=self.B, j=j)
                final_particles = smc.run()
                self.log_lambdas[j, :] = final_particles.reshape(-1)
            else:
                smc = SMC(Num=1, phi=self.Phi, sigma0=self.sigma0, y=self.Ay, z=self.z, B=self.B, j=j)
                final_particles = smc.run()
                self.log_lambdas[j, :self.T - 1] = final_particles.reshape(-1)[:self.T - 1]
                ####### does not take into the last time point

    def _rejuvenate_one_theta(self, num_iterations, tracker = None):

        #add tracking of resources
        if tracker is not None:
            tracker.append(threading.get_ident())  

        fix_log_lambda_phi_i = []
        df_phi = []

        for k in range(self.N):
            #idx = np.random.choice(self.N_x, p=weights[:, k])
            if k < self.k:
                df_phi.append(self.T)
                fix_log_lambda_phi_i.append(self.log_lambdas[k, :])
            else:
                df_phi.append(self.T - 1)
                fix_log_lambda_phi_i.append(self.log_lambdas[k, :self.T -1])

        A_i = self.A
        Pi_i = self.Pi
        Phi_i = self.Phi
        Ay = self.y @ A_i.T
        
        first_iter = True
        for _ in tqdm.tqdm(range(num_iterations)):
            #### here can pad the unseen data with zeros?
            # seems more convenient to pad log lambdas
            Pi_i = update_APi(
                Ay, self.z, Pi_i, self.log_lambdas, self.p,
                A_i, self.Pi_prior_mean, self.Pi_prior_var_inv, first_iter
            )
            first_iter = False # skip the rejection sampling for the initialization from prior
            Phi_i = compute_posterior_phi_by_component(
                fix_log_lambda_phi_i, df_phi, K=self.N
            )
            Api_i = A_i @ Pi_i


            fix_log_lambda_phi_i = []
            for k in range(self.N):
                t_k = self.T if k < self.k else self.T -1 ##### change here
                csmc = CSMC(
                Num= self.Num,
                phi=Phi_i,
                sigma0=self.sigma0,
                y=Ay[:t_k],
                z=self.z[:t_k],
                B=Api_i,
                j=k,
                fixed_particles=self.log_lambdas[k, :t_k]
                )
                _, _, sampled_traj, _, _ = csmc.run(if_reconstruct=True)
                
                fix_log_lambda_phi_i.append(sampled_traj)
                self.log_lambdas[k, :t_k] = sampled_traj
            
            # add the new rejuvenation of A
            APiz = A_i @ Pi_i @ self.z[self.T - 1]
            for k in range(self.k, self.N):
                #for k > self.current_k:
                self.log_lambdas[k, self.T - 1] = self.log_lambdas[k, self.T - 2] + np.sqrt(Phi_i[k]) * np.random.normal(0,1)
                # sampling unobserved tilde{y}, replace in placed in Ay
                Ay[self.T - 1, k] = np.random.normal(
                    APiz[k], 
                    #### correct the bug here!
                    np.exp(self.log_lambdas[k, self.T-1]) ** .5
                )

            syn_y = Ay @ np.linalg.inv(A_i.T)
            A_i = compute_posterior_A_with_log_lambdas(
                syn_y, self.z,
                self.log_lambdas, self.mu_A, self.Sigma_A, Pi_i
            )
            
            self.A = A_i
            self.Pi = Pi_i
            self.Phi = Phi_i
            self.Api= Api_i
            
            ##### debug here!!
            Ay = self.y @ A_i.T
            # renew 
            self.log_lambdas[self.k : self.N , self.T - 1] = np.full((self.N - self.k,), np.inf)

            # Store samples
            self.samples["A"].append(self.A.copy())
            self.samples["Pi"].append(self.Pi.copy())
            self.samples["log_lambdas"].append(self.log_lambdas.copy())
            self.samples["phi"].append(self.Phi.copy())



    def run(self, num_iterations):
        # Run the Gibbs sampler for a specified number of iteration
        self.initialize()
        self.samples = {
            "A": [],
            "Pi": [],
            "log_lambdas": [],
            'phi':[],
            'ancestors': [],
            'trejactories': []
        }

        self._rejuvenate_one_theta(num_iterations= num_iterations)

        return self.samples