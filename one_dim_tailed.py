import numpy as np

from matplotlib import pyplot as plt

from src.functions import \
    KF_step, KF_noise_step, MRobust_step, MRobust_noise_step

plt.style.use("seaborn-v0_8-darkgrid")

MAXSTEPS = 100
N = 25

# ------------------------------------------------------------------------------

T = 0.1

sigma_u = 0.01   # input to acceleration is random noise

sigma_a = 0.2
sigma_R = 5

F = np.array([[1, T, 0.5*T**2], [0, 1, T], [0, 0, 1]])
G = np.array([0.5*T**2, T, 1])
H = np.array([[1, 0, 0]])

Q = sigma_a**2*np.array(
    [
        [0.25*T**4, 0.5*T**3, 0.5*T**2],
        [0.5*T**3, T**2, T],
        [0.5*T**2, T, 1]
    ])

R = np.array([[sigma_R]])

x_init = np.array([0, 2, 0])

n = F.shape[0]
m = H.shape[0]

# ==============================================================================
#                             Generate trajectories
# ------------------------------------------------------------------------------
x = np.zeros((3, 1, MAXSTEPS))
x[:, 0, 0] = x_init
for k in range(1, MAXSTEPS):
    x[:, 0, k] = F.dot(x[:, 0, k-1]) + np.random.normal(loc=0, scale=sigma_u)*G

traj = x

normal_noise = np.random.normal(loc=0, scale=sigma_R, size=MAXSTEPS)
# normal_noise = np.random.normal(loc=0, scale=1, size=MAXSTEPS)

eps = 0.15

tailed_noise = (1 - eps)*np.random.normal(loc=0, scale=sigma_R, size=MAXSTEPS) \
                + eps*np.random.normal(loc=0, scale=4*sigma_R, size=MAXSTEPS)

# ==============================================================================
#                                   Filtering
# ------------------------------------------------------------------------------
P_init = 10*np.eye(3)
x_init = np.zeros(3) + 0.1

z_n = x[0, :, :] + normal_noise
z_t = x[0, :, :] + tailed_noise

# ------------------------------------------------------------------------------
#                                 Kalman filter
# ------------------------------------------------------------------------------
# Normal noise
x_kf_n = np.zeros((n, MAXSTEPS))
P_kf_n = np.zeros((n, n, MAXSTEPS))
K_kf_n = np.zeros((n, m, MAXSTEPS))

x = x_init
P = P_init

for k in range(0, MAXSTEPS):
    x, P, gain = KF_step(z_n[:, k], x, P, F, H, G, Q, R)
    x_kf_n[:, k], P_kf_n[:, :, k], K_kf_n[:, :, k] = x, P, gain

# Tailed noise
x_kf_t = np.zeros((n, MAXSTEPS))
P_kf_t = np.zeros((n, n, MAXSTEPS))
K_kf_t = np.zeros((n, m, MAXSTEPS))

x = x_init
P = P_init

for k in range(0, MAXSTEPS):
    x, P, gain = KF_step(z_t[:, k], x, P, F, H, G, Q, R)
    x_kf_t[:, k], P_kf_t[:, :, k], K_kf_t[:, :, k] = x, P, gain

cee_kf_n = np.cumsum(
    np.linalg.norm(np.array([x_kf_n[0, :] - traj[0, 0, :]]), axis=0) \
    / np.linalg.norm(traj[0, :])
)/np.arange(1, MAXSTEPS+1)

cee_kf_t = np.cumsum(
    np.linalg.norm(np.array([x_kf_t[0, :] - traj[0, 0, :]]), axis=0) \
    / np.linalg.norm(traj[0, :])
)/np.arange(1, MAXSTEPS+1)

# ------------------------------------------------------------------------------
#                         Kalman with noise estimation
# ------------------------------------------------------------------------------
# Normal noise
x_kfn_n = np.zeros((n, MAXSTEPS))
P_kfn_n = np.zeros((n, n, MAXSTEPS))
K_kfn_n = np.zeros((n, m, MAXSTEPS))

R_kfn_n = np.zeros((m, m, MAXSTEPS))
Q_kfn_n = np.zeros((n, n, MAXSTEPS))

r_kfn_n = np.zeros((n, MAXSTEPS))
q_kfn_n = np.zeros((m, MAXSTEPS))

x = x_init
P = P_init

R_e = R
Q_e = Q

r_mean = np.zeros((n, 1))
q_mean = np.zeros((m, 1))

for k in range(0, MAXSTEPS):
    if k < N:
        x_ests = x_kfn_n[:, :k]
        x_ests = np.insert(x_ests, 0, x_init, axis=1)
        z = z_n[:, :k+1]
    else:
        x_ests = x_kfn_n[k-N:k]
        z = z_n[:, k-N:k]
    
    P_prev = P

    x, P, gain, r_mean, q_mean, R_e, Q_e = KF_noise_step(
        z, x_ests, P, F, H, G, r_mean, q_mean, R_e, Q_e
    )
    x_kfn_n[:, k], P_kfn_n[:, :, k], K_kfn_n[:, :, k] = x, P, gain

    R_kfn_n[:, :, k], r_kfn_n[:, k] = R_e, r_mean.squeeze()
    Q_kfn_n[:, :, k], q_kfn_n[:, k] = Q_e, q_mean.squeeze()

# # Tailed noise
# x_mr_t = np.zeros((n, MAXSTEPS))
# P_mr_t = np.zeros((n, n, MAXSTEPS))
# K_mr_t = np.zeros((n, m, MAXSTEPS))

# x = x_init
# P = P_init

# for k in range(0, MAXSTEPS):
#     x, P, gain = MRobust_step(z_t[:, k], x, P, F, H, G, Q, R)
#     x_mr_t[:, k], P_mr_t[:, :, k], K_mr_t[:, :, k] = x, P, gain

# cee_mr_n = np.cumsum(
#     np.linalg.norm(np.array([x_mr_n[0, :] - traj[0, :]]), axis=0) \
#     / np.linalg.norm(traj[0, :])
# )/np.arange(1, MAXSTEPS+1)

# cee_mr_t = np.cumsum(
#     np.linalg.norm(np.array([x_mr_t[0, :] - traj[0, :]]), axis=0) \
#     / np.linalg.norm(traj[0, :])
# )/np.arange(1, MAXSTEPS+1)

# ------------------------------------------------------------------------------
#                                M-Robust filter
# ------------------------------------------------------------------------------
# Normal noise
x_mr_n = np.zeros((n, MAXSTEPS))
P_mr_n = np.zeros((n, n, MAXSTEPS))
K_mr_n = np.zeros((n, m, MAXSTEPS))

x = x_init
P = P_init

for k in range(0, MAXSTEPS):
    x, P, gain = MRobust_step(z_n[:, k], x, P, F, H, G, Q, R)
    x_mr_n[:, k], P_mr_n[:, :, k], K_mr_n[:, :, k] = x, P, gain

# Tailed noise
x_mr_t = np.zeros((n, MAXSTEPS))
P_mr_t = np.zeros((n, n, MAXSTEPS))
K_mr_t = np.zeros((n, m, MAXSTEPS))

x = x_init
P = P_init

for k in range(0, MAXSTEPS):
    x, P, gain = MRobust_step(z_t[:, k], x, P, F, H, G, Q, R)
    x_mr_t[:, k], P_mr_t[:, :, k], K_mr_t[:, :, k] = x, P, gain

cee_mr_n = np.cumsum(
    np.linalg.norm(np.array([x_mr_n[0, :] - traj[0, :]]), axis=0) \
    / np.linalg.norm(traj[0, :])
)/np.arange(1, MAXSTEPS+1)

cee_mr_t = np.cumsum(
    np.linalg.norm(np.array([x_mr_t[0, :] - traj[0, :]]), axis=0) \
    / np.linalg.norm(traj[0, :])
)/np.arange(1, MAXSTEPS+1)


# ==============================================================================
# Plot the trajectory
fig, ax = plt.subplots(3, 1, figsize=(12, 9))
ax[0].plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :], label="Position")
ax[1].plot(T*np.arange(0, MAXSTEPS), traj[1, 0, :], label="Velocity")
ax[2].plot(T*np.arange(0, MAXSTEPS), traj[2, 0, :], label="Acceleration")

for k in [0, 1, 2]:
    ax[k].set_xlabel("Time [s]")
    ax[k].legend()
ax[0].set_ylabel("Distance [m]")
ax[1].set_ylabel("Velocity [m/s]")
ax[2].set_ylabel("Acceleration [m/s^2]")

fig.suptitle("Generated trajectories for 1D case")
fig.tight_layout()
fig.show()

# ------------------------------------------------------------------------------
# Plot the measurements
fig = plt.figure(figsize=(12,6))

plt.plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :], label="True position", ls='-.')
plt.plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :] + normal_noise, label="Normal noise")
plt.plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :] + tailed_noise, label="Tailed noise")

plt.ylabel("Distance [m]")
plt.xlabel("Time [s]")
plt.legend()

fig.suptitle("Generated measurements for 1D case")
fig.tight_layout()
fig.show()

# ------------------------------------------------------------------------------
# Plot the estimations
# fig, ax = plt.subplots(3, 1, figsize=(12, 12))
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :], label="True position", ls='-.')
ax[0].plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :] + normal_noise, label="Measurement")
ax[0].plot(T*np.arange(0, MAXSTEPS), x_kf_n[0, :], label="KF estimate")
ax[0].plot(T*np.arange(0, MAXSTEPS), x_mr_n[0, :], label="MR estimate")
ax[0].set_title("Normal noise")

ax[1].plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :], label="True position", ls='-.')
ax[1].plot(T*np.arange(0, MAXSTEPS), traj[0, 0, :] + tailed_noise, label="Measurement")
ax[1].plot(T*np.arange(0, MAXSTEPS), x_kf_t[0, :], label="KF estimate")
ax[1].plot(T*np.arange(0, MAXSTEPS), x_mr_n[0, :], label="MR estimate")
ax[1].set_title("Tailed noise")

for k in [0, 1]:#, 2]:
    ax[k].set_xlabel("Time [s]")
    ax[k].legend()
    ax[k].set_ylabel("Distance [m]")

fig.suptitle("Kalman fitler output")
fig.tight_layout()
fig.show()

# ------------------------------------------------------------------------------
# Plot the CEE metrics

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].plot(T*np.arange(0, MAXSTEPS), cee_kf_n, label="Kalman Filter")
ax[0].plot(T*np.arange(0, MAXSTEPS), cee_mr_n, label="M-Robust Filter")
ax[0].set_title("Normal noise")

ax[1].plot(T*np.arange(0, MAXSTEPS), cee_kf_t, label="Kalman Filter")
ax[1].plot(T*np.arange(0, MAXSTEPS), cee_mr_t, label="M-Robust Filter")
ax[1].set_title("Tailed noise")

fig.suptitle("CEE comparison")
fig.tight_layout()
fig.show()
# ==============================================================================
input()