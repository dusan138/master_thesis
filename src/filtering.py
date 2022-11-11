import numpy as np

def kalman_filter(F, H, Q, R, z, x_init, P_init):

    num_meas = z.shape[1]
    n = x_init.shape[0]
    m = z.shape[0]

    x_est = np.zeros((n, num_meas + 1))
    x_pred = np.zeros((n, num_meas))
    P_est = np.zeros((n, n, num_meas + 1))
    P_pred = np.zeros((n, n, num_meas))
    K = np.zeros((n, m, num_meas))

    x_est[:, 0] = x_init
    P_est[:, :, 0] = P_init

    z = z.T

    for k in range(0, num_meas):
        
        # Prediction
        x_pred[:, k] = F.dot(x_est[:, k])
        P_pred[:, :, k] = F.dot(P_est[:, :, k].dot(F.T)) + Q

        res = z[k] - H.dot(x_pred[:, k])
        res_cov = H.dot(P_pred[:, :, k].dot(H.T)) + R

        # Update
        K[:, :, k] = P_pred[:, :, k].dot(H.T.dot(np.linalg.inv(res_cov)))

        x_est[:, k+1] = x_pred[:, k] + K[:, :, k].dot(res)
        P_est[:, :, k+1] = (np.eye(n) - K[:, :, k].dot(H)).dot(P_pred[:, :, k])

    return x_est, P_est, K, x_pred, P_pred