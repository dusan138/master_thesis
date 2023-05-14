import numpy as np

def gen_traj(F, G, x_init=None, steps=100, acc_profile=None):
    """
    Generate trajectories.

    This function generates the coordinate values of a moving target.
    """

    n = F.shape[0]

    x = np.zeros((n, steps))

    x[:, 0] = np.zeros((n,)) if x_init is None else x_init

    for k in range(1, steps):
        x[:, k] = np.dot(F, x[:, k-1])

        if acc_profile is not None:
            x[:, k] += np.dot(G, acc_profile[k])

    return x

def gen_measurements(
    traj, H, R, change_points=[0], r_mean=None, tailed=False, epsilon=.15):
    """
    Simulate measurements from a given trajectory. 
    """

    n = H.shape[1]
    m = H.shape[0]
    
    steps = traj.shape[1]
    
    # Assume that noise changes at predefined points in `change_points` vector.
    
    num_changes = len(change_points)
    idx = 0

    noise_cov = R[:, :, 0]

    if r_mean is None:
        noise_mean = np.zeros((n,))
    else:
        noise_mean = r_mean[:, 0]

    z = np.zeros((m, steps))

    for k in range(steps):
        if (idx+1 < num_changes):
            if (k != 0) and (k % change_points[idx+1] == 0):
                idx += 1

                noise_cov = R[:, :, idx]
                if r_mean is not None:
                    noise_mean = r_mean[:, idx]

        if not tailed:
            z[:, k] = np.dot(H, traj[:, k]) \
                + np.random.multivariate_normal(mean=noise_mean, cov=noise_cov)
        else:
            pass

    return z

def add_clutter_noise(x_lim, x_div, y_lim, y_div, p_fa):
    """
    Simulate clutter noise in already generated measurements.
    """
    pass