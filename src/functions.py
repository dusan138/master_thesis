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

def MRobust_update(z, x_pred, x_est, P_pred, H, R, delta=1.5):
    """
    One step update for M robust filter.

    x_est: estimation of x from last step
    """

    # Assume Huber M-estimator
    psi = lambda z: delta*np.sign(z) if np.abs(z) >= delta else z

    n = x_pred.shape[0]
    m = z.shape[0]

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
    temp_mat = np.concatenate([np.eye(n), H])
    X = np.dot(np.linalg.inv(S), temp_mat)

    # Calculate the Y matrix
    temp_mat = np.concatenate([x_pred, z])
    Y = np.dot(np.linalg.inv(S), temp_mat)

    # Create the weight matrix
    W = np.eye(m + n)

    for k in range(0, m + n):
        arg = Y[k] - np.dot(X[k, :], x_est)

        W[k, k] = psi(arg)/arg if arg != 0 else 1

    # Update the estimate
    x_est = np.dot(
        np.linalg.pinv(np.dot(X.T, np.dot(W, X))),
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

    N = r.shape[1]

    d = np.median(np.abs(r - np.median(r))/robust_divisor)

    if r_mean is not None:
        # Calculate the weight matrix
        W = np.eye(N)

        for k in range(N):
            arg = ((r[:, k] - r_mean)/d)[0]

            W[k, k] = psi(arg)/arg if arg != 0 else 1

        r_mean = np.sum(np.dot(W, r.T))/np.trace(W)
    else:
        # None means we assume zero mean noise
        r_mean = 0

    cov_num = 0
    cov_div = 0

    for k in range(N):
        arg = (r[:, k] - r_mean)/d

        cov_num += psi(arg)**2
        cov_div += psi_der(arg)

    cov_num = (d**2)*cov_num/N
    cov_div = (cov_div/N)**2

    C_r = cov_num/cov_div

    return r_mean, C_r

def process_noise_statistics(q_mean, x_ests, F, G, P, P_prev, N=25):
    """
    Calculate mean and covariance of w noise.

    x_ests is vector of N + 1 last estimates
    """

    # Make sure we at most N samples in x_estimates
    x_ests = x_ests[:, -N-1:]

    tmp = np.dot(G.T, G)
    if type(tmp) == np.float64:
        tmp = np.array([[tmp]])

    # Approximate the noise
    q = np.dot(
        np.linalg.inv(tmp).squeeze(),
        np.dot(
            G.T, 
            x_ests[:, 1:]-np.dot(F, x_ests[:, :-1])
        )
    ) # this works, but just to be 100% sure, won't use it
    q = np.expand_dims(q, axis=0)

    # q = np.zeros(N)
    # for k in range(1, N+1):
    #     q[k] = np.dot(
    #         np.linalg.inv(tmp), 
    #         np.dot(G.T, x_ests[k-1] - np.dot(F, x_ests[k]))
    #     )

    q_mean, C_q = noise_statistics(q, q_mean)

    w_mean = q_mean

    # Q = C_q - np.dot(np.linalg.inv(tmp).squeeze(), np.dot(F, np.dot(P_prev, F.T)) - P)

    return w_mean.squeeze(), C_q.squeeze()

def measurement_noise_statistics(z, r_mean, x_ests, H, N=25):
    """
    Calculate mean and covariance of v noise.

    x_ests is vector of N last estimates!
    """

    # Make sure we at most N samples in x_estimates
    x_ests = x_ests[:, -N:]

    # Approximate the noise
    r = z - np.dot(H, x_ests)

    r_mean, C_r = noise_statistics(r, r_mean)

    v_mean = r_mean

    return v_mean.squeeze(), C_r.squeeze()

# ==============================================================================
#                                Filtering steps
# ------------------------------------------------------------------------------

def KF_step(z, x_est, P_est, F, H, G, Q, R, u=0):
    """
    Kalman filter estimation step.
    """

    x_pred, P_pred = KF_predict(x_est, P_est, F, G, Q, u)

    x_est, P_est, gain = KF_update(z, x_pred, P_pred, H, R)

    return x_est, P_est, gain

def KF_noise_step(
        z, x_ests, P_est, F, H, G, r_mean, q_mean, R, Q, N=25, u=0):
    """
    Kalman filter with robust noise estimation step
    """

    P_est_prev = P_est

    x_est = x_ests[:, -1]
    meas = z[:, -1]

    x_pred, P_pred = KF_predict(x_est, P_est, F, G, Q)

    x_est, P_est, gain = KF_update(meas, x_pred, P_pred, H, R)

    if x_ests.shape[1] > 2:
        # Estimate statistics only if there's enough data for that

        # Go through dimensions of x and assume that they're independent! 
        # Estimate the noise characteristics
        # Assume here that all the coordinates in the state vector are independent
        # => this means that the noise covariances R and Q are diagonal and each of
        # the paramters can be estimated separately
        r_mean_new = np.zeros(R.shape[0])
        C_r = np.zeros(R.shape)
        for dim in range(R.shape[0]):
            r_mean_new[dim], C_r[dim, dim] = \
                measurement_noise_statistics(z, r_mean[dim], x_ests, H)

        R = C_r - np.dot(H, np.dot(P_est, H.T))

        q_mean_new = np.zeros(Q.shape[0])
        C_q = np.zeros(Q.shape)
        for dim in range(Q.shape[0]):
            q_mean_new[dim], C_q[dim, dim] = \
                process_noise_statistics(
                    q_mean[dim], x_ests, F, G, P_est, P_est_prev)
        
        Q = C_q
    else:
        r_mean_new = r_mean
        q_mean_new = q_mean

    return x_est, P_est, gain, r_mean_new, q_mean_new, R, Q

def MRobust_step(z, x_est, P_est, F, H, G, Q, R, u=0):
    """
    M-robust filter without noise estimation step.
    """
    
    x_pred, P_pred = KF_predict(x_est, P_est, F, G, Q, u)

    x_est, P_est, gain = MRobust_update(z, x_pred, x_est, P_pred, H, R)

    return x_est, P_est, gain

def MRobust_noise_step(
        z, x_ests, P_est, P_est_prev, F, H, G, r_mean, q_mean, R, Q, N=25, u=0):
    """
    M-robust filter with noise estimation step.
    """

    x_est = x_ests[:, -1]

    x_pred, P_pred = KF_predict(x_est, P_est, F, G, Q, u)

    x_est, P_est, gain = MRobust_update(z, x_pred, x_est, P_pred, H, R)

    # Go through dimensions of x and assume that they're independent! 
    # Estimate the noise characteristics
    # Assume here that all the coordinates in the state vector are independent
    # => this means that the noise covariances R and Q are diagonal and each of
    # the paramters can be estimated separately
    r_mean_new = np.zeros(R.shape[0])
    for dim in range(R.shape[0]):
        r_mean_new[dim], R[dim, dim] = \
            measurement_noise_statistics(z, r_mean, x_ests, H, P_est)

    q_mean_new = np.zeros(Q.shape[0])
    for dim in range(Q.shape[0]):
        q_mean_new[dim], Q[dim, dim] = \
            process_noise_statistics(q_mean, x_ests, F, G, P_est, P_est_prev)

    return x_est, P_est, gain, r_mean_new, R, q_mean_new, Q