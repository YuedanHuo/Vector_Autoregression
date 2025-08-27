from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np

class BayesianVARPrior:
    def __init__(self, y, p=1, sigma_0=2.31, threshold=0.5):
        """
        Class to set up the prior for the Bayesian VAR model.

        Parameters:
        y (numpy.ndarray): The time series data (T, N), where T is the number of observations and N is the number of variables.
        p (int): The lag order of the AR model (default is 1).
        sigma_0 (float): The scaling factor for the prior covariance (default is 2.31).
        threshold (float): The threshold for classifying high vs. low persistence (default is 0.5).
        """
        self.y = y
        self.p = p
        self.sigma_0 = sigma_0
        self.threshold = threshold
        self.T, self.N = y.shape


    def compute_persistence(self):
        """
        Determines the persistence of variables based on lag-1 autocorrelation.
        """
        autocorrs = np.zeros(self.N)

        for i in range(self.N):
            # Compute lag-1 autocorrelation for each variable
            autocorrs[i] = pd.Series(self.y[:, i]).autocorr(lag=1)

        # Classify persistence: 1 for high persistence, 0 for low persistence
        persistence = (autocorrs > self.threshold).astype(int)

        # Construct diagonal matrix
        Pi_1 = np.diag(persistence)

        return Pi_1, autocorrs

    def compute_sigma_i_squared(self):
        """
        Computes sigma_i^2 for each variable in the dataset using a univariate AR(p) model.
        """
        sigma_squared = np.zeros(self.N)

        for i in range(self.N):
            # Fit AR(p) model for the i-th variable
            model = AutoReg(self.y[:, i], lags=self.p, old_names=False).fit()
            # Compute residual variance
            residuals = model.resid
            sigma_squared[i] = np.var(residuals, ddof=1)  # Unbiased variance estimate

        return sigma_squared

    def setup_Pi_prior_mean(self):
        """
        Set up Pi prior mean matrix.
        """
        Pi_prior_mean = np.zeros((self.N, self.N * self.p + 1))
        Pi_prior_mean[:, 1:self.N + 1] = self.Pi_1_prior_mean
        return Pi_prior_mean

    def setup_Pi_prior_var(self):
        """
        Set up the prior variance for the VAR coefficients.
        """
        Pi_prior_var_rows = []

        for i in range(self.N):
            Pi_prior_var_row = np.zeros((self.N * self.p + 1, self.N * self.p + 1))

            # Set the prior variance for intercept (first entry in the row)
            Pi_prior_var_row[0, 0] = 100  # Prior for intercepts

            for lag in range(self.p):
                for j in range(self.N):
                    index = lag * self.N + j + 1  # Compute correct index for the lag and variable

                    if i == j:
                        Pi_prior_var_row[index, index] = 0.05 / (lag + 1) ** 2
                    else:
                        Pi_prior_var_row[index, index] = (
                            0.05 * 0.5 * self.sigma_squared[i] / (self.sigma_squared[j] * (lag + 1) ** 2)
                        )

            Pi_prior_var_rows.append(Pi_prior_var_row)

        # Calculate the inverse of covariance matrices
        Pi_prior_var_rows_inv = [np.linalg.inv(row) for row in Pi_prior_var_rows]
        return Pi_prior_var_rows, Pi_prior_var_rows_inv

    def construct_Z_matrix(self):
        """
        Constructs the matrix Z with intercepts and lagged values of variables.
        """
        T, N = self.y.shape
        Z = np.ones((T - self.p, N * self.p + 1))  # Include a column of 1's for intercept

        # Fill in the lagged values
        for lag in range(1, self.p + 1):
            Z[:, 1 + (lag - 1) * N: 1 + lag * N] = self.y[self.p - lag:T - lag, :]

        # Trim the dependent variable to match the lagged matrix
        y_trimmed = self.y[self.p:, :]
        self.T, self.N = y_trimmed.shape

        return Z, y_trimmed

    def get_priors(self):
        """
        Returns all priors: Pi_1, sigma_i^2, Pi_prior_mean, Pi_prior_var
        """

        self.Z, self.y = self.construct_Z_matrix()

        self.sigma_squared = self.compute_sigma_i_squared()
        self.Pi_1_prior_mean, _ = self.compute_persistence()
        self.Pi_prior_mean = self.setup_Pi_prior_mean()
        Pi_prior_var_rows, Pi_prior_var_rows_inv = self.setup_Pi_prior_var()


        # Phi (sigma_i^2), sigma_0, and prior matrices
        phi = self.sigma_squared
        sigma_0 = self.sigma_0
        mu_A = np.zeros(self.N) # we impose the same prior on each row of A, and all of them iid
        Sigma_A = np.eye(self.N)*10**6

        return {
            "phi": phi,
            "Pi_prior_mean": self.Pi_prior_mean,
            'Pi_prior_var': Pi_prior_var_rows,
            "Pi_prior_var_inv": Pi_prior_var_rows_inv,
            "Sigma_A": Sigma_A,
            "mu_A": mu_A,
            "Z": self.Z,
            "y": self.y,
            'sigma_0': sigma_0,
            # use this to be the long run mean of AR(1) process of volatility
            'mu_volatility': self.sigma_squared,
        }

