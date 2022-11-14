import numpy as np


class KalmanFilter():

    def __init__(self, F, H, Q, R, G=None):

        self.n = F.shape[1]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.G = 0 if G is None else G

    def predict(self, x_est, P_est, u=0):

        x_pred = self.F.dot(x_est) + np.dot(self.G, u)
        P_pred = self.F.dot(P_est).dot(self.F.T) + self.Q

        return x_pred, P_pred

    def update(self, z, x_pred, P_pred):

        res = z - self.H.dot(x_pred)
        res_cov = self.H.dot(P_pred).dot(self.H.T) + self.R

        gain = P_pred.dot(self.H.T).dot(np.linalg.inv(res_cov))

        x_est = x_pred + gain.dot(res)
        P_est = (np.eye(self.n) - gain.dot(self.H)).dot(P_pred)

        return x_est, P_est, gain

    def step(self, z, x_est, P_est):

        x_pred, P_pred = self.predict(x_est, P_est)
        x_est, P_est, gain = self.update(z, x_pred, P_pred)

        return x_est, P_est, gain

    def estimate(self, meas, x_init, P_init):

        steps = meas.shape[1]

        x_est = np.zeros((self.n, steps))
        P_est = np.zeros((self.n, self.n, steps))
        gains = np.zeros((self.n, self.m, steps))

        x = x_init
        P = P_init

        for k in range(0, steps):
            z = meas[:, k]
            x, P, gain = self.step(z, x, P)
            x_est[:, k], P_est[:, :, k], gains[:, :, k] = x, P, gain

        return x_est, P_est, gains
