import numpy as np
from collections import deque as stack

# ==============================================================================
#                                 Kalman filter
# ------------------------------------------------------------------------------

def KF_predict(x_est, P_est, F, G, Q, u=0):
    """
    One step prediction for Kalman filter.
    """

    x_pred = np.dot(F, x_est) + np.dot(G, u).squeeze()
    P_pred = np.dot(F, np.dot(P_est, F.T)) + Q

    return x_pred, P_pred

def KF_update(z, x_pred, P_pred, H, R):
    """
    One step update for Kalman filter.
    """

    n = x_pred.shape[0]

    res = z - np.dot(H, x_pred)
    res_cov = np.dot(H, np.dot(P_pred, H.T)) + R

    if type(res_cov) == np.float64:
        res_cov = np.array([[res_cov]])

    gain = np.dot(np.dot(P_pred, H.T), np.linalg.inv(res_cov))

    x_est = x_pred + np.dot(gain, res)
    P_est = np.dot((np.eye(n) - np.dot(gain, H)), P_pred)

    return x_est, P_est, gain


# ==============================================================================
#                              M-robust estimation
# ------------------------------------------------------------------------------

def MRobust_update(
        z, x_pred, x_est, P_pred, H, R, delta=1.5, robust_divisor=0.6745):
    """
    One step update for M robust filter.

    x_est: estimation of x from last step
    """

    # Assume Huber M-estimator
    psi = lambda z: delta*np.sign(z) if np.abs(z) >= delta else z
    psi_der = lambda z: 0 if np.abs(z) >= delta else 1

    n = x_pred.shape[0]
    m = z.shape[0]

    res = z - np.dot(H, x_pred)
    res_cov = np.dot(H, np.dot(P_pred, H.T)) + R

    if type(res_cov) == np.float64:
        res_cov = np.array([[res_cov]])

    gain = np.dot(np.dot(P_pred, H.T), np.linalg.inv(res_cov))

    # Calculate the S matrix by Cholesky decomposition
    temp_mat = np.concatenate(
        [np.concatenate([P_pred, np.zeros((m, n))]),
            np.concatenate([np.zeros((n, m)), R])],
        axis=1
    )

    S = np.linalg.cholesky(temp_mat)

    # Calculate the X matrix
    temp_mat = np.concatenate([np.eye(m), H])
    X = np.dot(np.linalg.inv(S), temp_mat)

    # Calculate the Y matrix
    temp_mat = np.concatenate([x_pred, z])
    Y = np.dot(np.linalg.inv(S), temp_mat)

    # Create the weight matrix
    W = np.eye(m + n)

    for k in range(0, m + n):
        W[k, k] = psi(Y[:, k] - np.dot(X[:, k], x_est))

    # Update the estimate
    x_est = np.dot(
        np.linalg.inv(np.dot(X.T, np.dot(W, X))),
        np.dot(X.T, np.dot(W, Y))
    )
    P_est = np.dot((np.eye(n) - np.dot(gain, H)), P_pred)

    return x_est, P_est, gain


# ==============================================================================
#                           M-robust noise estimation
# ------------------------------------------------------------------------------

def noise_statistics(r, r_mean=None, delta=1.5, robust_divisor=0.6745):
    """
    Calculate mean and covariance of noise.
    """

    # Assume Huber M-estimator
    psi = lambda z: delta*np.sign(z) if np.abs(z) >= delta else z
    psi_der = lambda z: 0 if np.abs(z) >= delta else 1

    N = r.shape[0]

    d = np.median(np.abs(r - np.median(r))/robust_divisor)

    if r_mean is not None:
        # Calculate the weight matrix
        W = np.eye(N)

        for k in range(N):
            W[k, k] = psi((r[k] - r_mean)/d)
    else:
        # None means we assume zero mean noise
        r_mean = 0

    cov_num = 0
    cov_div = 0

    for k in range(N):
        arg = (r[k] - r_mean)/d

        cov_num += psi(arg)**2
        cov_div += psi_der(arg)

    cov_num = (d**2)*cov_num/N
    cov_div = (cov_div/N)**2

    C_r = cov_num/cov_div

    return r_mean, C_r

def process_noise_statistics():
    """
    Calculate mean and covariance of w noise.
    """
    pass

def measurement_noise_statistics(z, x_estimates, H, N=25):
    """
    Calculate mean and covariance of v noise.

    x_estimates is vector of N last estimates!
    """

    # Approximate the noise
    r = z - np.dot(H, x_estimates)


# ==============================================================================
#                                Filtering steps
# ------------------------------------------------------------------------------

def KF_step(z, x_est, P_est, F, H, G, Q, R, u=0):
    """
    Kalman filter estimation step.
    """

    x_pred, P_pred = KF_predict(x_est, P_est, F, G, Q, u)

    x_est, P_est, gain = KF_update(z, x_pred, P_pred, H, Q)

    return x_est, P_est, gain