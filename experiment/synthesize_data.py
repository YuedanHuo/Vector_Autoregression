import numpy as np

def simulate_sv_var(T, K, p, Pi=None, A_inv=None, Phi=None, seed=0):
    """
    Simulate data from the SV-VAR(p) model.

    Args:
        T (int): Number of time steps
        K (int): Number of variables
        p (int): VAR order
        Pi (np.ndarray): Optional, shape (K, K*p + 1)
        A_inv (np.ndarray): Optional, shape (K, K), lower triangular with 1s on diagonal
        Phi (np.ndarray): Optional, shape (K,), diagonal of volatility process
        seed (int): Random seed

    Returns:
        y (np.ndarray): shape (T, K)
        log_lambda (np.ndarray): shape (T, K)
        z (np.ndarray): shape (T, K*p + 1)
    """
    np.random.seed(seed)

    # Dimensions
    z_dim = K * p + 1

    # Set default parameters if not provided
    if Pi is None:
        Pi = np.random.randn(K, z_dim) * 0.1

    if A_inv is None:
        A_inv = np.tril(np.random.randn(K, K) * 0.1)
        np.fill_diagonal(A_inv, 1.0)

    if Phi is None:
        Phi = 0.01 * np.ones(K)

    # Initialize storage
    y = np.zeros((T+p, K))
    log_lambda = np.zeros((T+p, K))
    z_hist = np.zeros((T+p, z_dim))  # Stores z_t = [1, y_{t-1}, ..., y_{t-p}]

    # Initial values
    log_lambda[0] = np.random.randn(K) * 0.1
    for t in range(1, T+p):
        log_lambda[t] = log_lambda[t - 1] + np.random.randn(K) * np.sqrt(Phi)

    for t in range(p, T+p):
        # Build z_t = [1, y_{t-1}^T, ..., y_{t-p}^T]
        z_t = [np.array([1.0])]
        for lag in range(1, p + 1):
            z_t.append(y[t - lag])
        z_t = np.concatenate(z_t)
        z_hist[t] = z_t

        # Compute Lambda_t
        Lambda_t = np.diag(np.exp(log_lambda[t]))

        # Sample epsilon ~ N(0, I_K)
        eps = np.random.randn(K)

        # Sample v_t = A^{-1} Λ^{1/2} ε
        v_t = A_inv @ np.sqrt(Lambda_t) @ eps

        # y_t = Π z_t + v_t
        y[t] = Pi @ z_t + v_t

    return y[p:], log_lambda[p:], z_hist[p:], Pi, A_inv, Phi

import numpy as np

def simulate_sv_var_stable(T, K, p, Pi=None, A_inv=None, Phi=None, seed=0):
    """
    Simulate data from a stable SV-VAR(p) model with stationary log-volatilities.

    Args:
        T (int): Number of time steps.
        K (int): Number of variables.
        p (int): VAR order.
        Pi (np.ndarray): shape (K, K*p + 1), optional.
        A_inv (np.ndarray): shape (K, K), optional, lower triangular with 1s on diagonal.
        Phi (np.ndarray): shape (K,), optional, volatility innovation variance.
        seed (int): Random seed.

    Returns:
        y (np.ndarray): shape (T, K)
        log_lambda (np.ndarray): shape (T, K)
        z_hist (np.ndarray): shape (T, K*p + 1)
        Pi, A_inv, Phi: true parameter values used
    """
    np.random.seed(seed)

    z_dim = K * p + 1

    # Set defaults if not provided
    if Pi is None:
        Pi = np.random.randn(K, z_dim) * 0.05  # small coefficients for stability

    if A_inv is None:
        A_inv = np.tril(np.random.randn(K, K) * 0.05)
        np.fill_diagonal(A_inv, 1.0)

    if Phi is None:
        Phi = 0.05 * np.ones(K)  # variance of log-volatility innovations

    sigma_eta = np.sqrt(Phi)

    # Storage
    y = np.zeros((T + p, K))
    log_lambda = np.zeros((T + p, K))
    z_hist = np.zeros((T + p, z_dim))

    # Initialize log_lambda with stationary AR(1)
    log_lambda[0] = np.random.normal(0, np.sqrt(2.31),K)
    for t in range(1, T + p):
        log_lambda[t] = log_lambda[t - 1] + np.random.normal(K) * sigma_eta
        #log_lambda[t] = np.clip(log_lambda[t], -10, 2)  # avoid explosion

    for t in range(p, T + p):
        # z_t = [1, y_{t-1}^T, ..., y_{t-p}^T]
        z_t = [np.array([1.0])]
        for lag in range(1, p + 1):
            z_t.append(y[t - lag])
        z_t = np.concatenate(z_t)
        z_hist[t] = z_t

        # Lambda_t = diag(exp(log_lambda))
        Lambda_t = np.diag(np.exp(log_lambda[t]))

        # ε ~ N(0, I_K)
        eps = np.random.randn(K)

        # v_t = A^{-1} sqrt(Lambda_t) ε
        v_t = A_inv @ np.sqrt(Lambda_t) @ eps

        # y_t = Pi z_t + v_t
        y[t] = Pi @ z_t + v_t

    return y[p:], log_lambda[p:], z_hist[p:], Pi, A_inv, Phi
