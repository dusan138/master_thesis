from abc import ABC, abstractmethod

import numpy as np
from collections import deque as stack


class Algorithm(ABC):

    def __init__(self, F, H, Q=None, R=None, G=None):

        self.n = F.shape[1]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        if Q is not None:
            self.Q = Q if G is None else G.dot(G.T)*Q
        if R is not None:
            self.R = R
        self.G = 0 if G is None else G

    @abstractmethod
    def predict(self):

        pass

    @abstractmethod
    def update(self):

        pass

    @abstractmethod
    def step(self):

        pass

    @abstractmethod
    def estimate(self):

        pass


class Algorithm_1(Algorithm):
    """
    Implementation of Kalman filter
    """
    
    def __init__(self, F, H, Q, R, G=None):

        super(Algorithm_1, self).__init__(F, H, Q, R, G)

    def predict(self, x_est, P_est, u=0):
        """
        One step prediction for Kalman filter.
        """

        x_pred = self.F.dot(x_est) + np.dot(self.G, u).squeeze()
        P_pred = self.F.dot(P_est).dot(self.F.T) + self.Q

        return x_pred, P_pred

    def update(self, z, x_pred, P_pred):
        """
        One step update for Kalman filter.
        """

        res = z - np.dot(self.H, x_pred)
        res_cov = np.dot(self.H, np.dot(P_pred, self.H.T)) + self.R

        if type(res_cov) == np.float64:
            res_cov = np.array([[res_cov]])

        gain = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(res_cov))

        x_est = x_pred + np.dot(gain, res)
        P_est = np.dot((np.eye(self.n) - np.dot(gain, self.H)), P_pred)

        return x_est, P_est, gain

    def step(self, z, x_est, P_est):
        """
        One estimation step for Kalman filter.
        """
        x_pred, P_pred = self.predict(x_est, P_est)
        x_est, P_est, gain = self.update(z, x_pred, P_pred)

        return x_est, P_est, gain

    def estimate(self, meas, x_init, P_init):
        """
        Estimation procedure from a vector of measurements.
        """
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


class Algorithm_2(Algorithm):
    """
    Implementation of Kalman filter with robust noise estimation
    """
    def __init__(self, F, H, N, G=None, robust_divisor=0.6745, delta=1.5):

        super(Algorithm_2, self).__init__(F, H, G)

        self.robust_divisor = robust_divisor
        self.delta = delta

        self.psi = \
            lambda z: self.delta*np.sign(z) if np.abs(z) > self.delta else z
        self.psi_der = lambda z: 0 if np.abs(z) > self.delta else 1

        self.N = N

    def __noise_statistics(
            self,
            r,
            P_pred,
            P_prev=None,
            noise_type="measurement",
            r_mean_prev=None):

        N = r.shape[0]

        d = np.median(np.abs(r - np.median(r))/self.robust_divisor)

        if r_mean_prev is not None:
            # Calculate new mean estimate if previous mean estimate is provided
            W = np.eye(N)

            for k in range(N):
                W[k, k] = self.psi((r[k] - r_mean_prev)/d)

            mean_est = np.sum(np.dot(W, np.transpose(r))) / np.trace(W)
        else:
            mean_est = 0

        num = 0
        div = 0

        for k in range(N):
            arg = (r[k] - mean_est)/d
            num += self.psi(arg)**2
            div += self.psi_der(arg)

        num = num/N
        div = (1/N*div)**2

        C = d**2*num/div

        if noise_type == "measurement":
            var_est = C - np.dot(self.H, np.dot(P_pred, self.H.T))
        elif noise_type == "process":
            var_est = C - \
                np.dot(
                    np.linalg.inv(np.dot(np.transpose(self.G), self.G)),
                    np.dot(self.F, np.dot(P_prev, self.F.T)) - P_pred
                )
        
        return mean_est, var_est

    def predict(self, x_est, P_est, Q, u=0):
        """
        One step prediction for Kalman filter.
        """

        x_pred = self.F.dot(x_est) + np.dot(self.G, u)
        P_pred = self.F.dot(P_est).dot(self.F.T) + Q

        return x_pred, P_pred

    def update(self, z, x_pred, P_pred, R):
        """
        One step update for Kalman filter with M-robust noise estimate
        """

        res = z - np.dot(self.H, x_pred)
        res_cov = np.dot(self.H, np.dot(P_pred, self.H.T)) + R

        gain = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(res_cov))

        x_est = x_pred + np.dot(gain, res)
        P_est = np.dot((np.eye(self.n) - np.dot(gain, self.H)), P_pred)

        return x_est, P_est, gain

    def step(
            self, z, x_est, P_est,
            P_pred_prev,
            r, q,
            R, Q,
            r_mean, q_mean):

        x_pred, P_pred = self.predict(x_est, P_est, Q)
        x_est, P_est, gain = self.update(z, x_pred, P_pred, R)
        # Estimate the noise characteristics
        # We are assuming that all the coordinates in the vector are independent
        # => this means that the noise covariances are diagonal, and we can 
        # estimate each parameter separately
        for k in range(R.shape[0]):
            r_mean[k], R[k, k] = self.__noise_statistics(
                r=r,
                P_pred=P_pred,
                P_prev=P_pred_prev,
                noise_type="measurement",
                r_mean_prev=r_mean[k]
            )

        for k in range(Q.shape[0]):
            q_mean[k], Q[k, k] = self.__noise_statistics(
                r=q,
                P_pred=P_pred,
                P_prev=P_pred_prev,
                noise_type="process",
                r_mean_prev=q_mean[k]
            )

        return x_est, P_est, gain, r_mean, R, q_mean, Q

    def estimate(self, meas, x_init, P_init):
        # Create stacks for r and q 
        steps = meas.shape[1]

        x_est = np.zeros((self.n, steps))
        P_est = np.zeros((self.n, self.n, steps))
        gains = np.zeros((self.n, self.m, steps))
        r_mean = np.zeros((self.m, steps))
        q_mean = np.zeros((self.n, steps))
        R = np.zeros((self.m, self.m, steps))
        Q = np.zeros((self.n, self.n, steps))

        r = stack()
        q = stack()

        x = x_init
        x_prev = x_init
        P = P_init
        P_prev = P_init

        for k in range(0, steps):
            
            z = meas[:, k]

            r.append(z - np.dot(self.H, x))
            q.append(
                np.dot(
                    np.linalg.inv(np.dot(np.transpose(self.G), self.G)),
                    np.dot(self.G, (x - np.dot(self.F, x_prev)))
                )
            )

            x_prev = x
            P_prev = P

            if len(r) > self.N:
                r.popleft()
            if len(q) > self.N:
                q.popleft()


class Algorithm_3(Algorithm):
    """
    Implementation of M Robust estimate filter
    """

    def __init__(self, F, H, Q, R, G=None):

        super(Algorithm_3, self).__init__(F, H, Q, R, G)

        self.psi = \
            lambda z: self.delta*np.sign(z) if np.abs(z) > self.delta else z
        self.psi_der = lambda z: 0 if np.abs(z) > self.delta else 1

    def predict(self, x_est, P_est, u=0):
        """
        One step prediction for M Robust estimate filter.
        """

        x_pred = self.F.dot(x_est) + np.dot(self.G, u)
        P_pred = self.F.dot(P_est).dot(self.F.T) + self.Q

        return x_pred, P_pred
    
    def update(self, z, x_pred, x_est_prev, P_pred):
        """
        One step update for Kalman filter.
        """

        res_cov = np.dot(self.H, np.dot(P_pred, self.H.T)) + self.R

        gain = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(res_cov))

        temp_mat = np.concatenate(
            [np.concatenate([P_pred, np.zeros((self.m, self.n))]),
             np.concatenate([np.zeros((self.n, self.m)), self.R])],
            axis=1
        )

        S = np.linalg.cholesky(temp_mat)

        temp_mat = np.concatenate([np.eye(self.m), self.H])
        X = np.dot(np.linalg.inv(S), temp_mat)

        temp_mat = np.concatenate([x_pred, z])
        Y = np.linalg.inv(S).dot(temp_mat)

        W = np.eye(self.m + self.n)

        for k in range(0, self.m + self.n):
            W[k, k] = self.psi(Y[:, k] - np.dot(X[:, k], x_est_prev))

        x_est = np.dot(
            np.linalg.inv(np.dot(X.T, np.dot(W, X))),
            np.dot(X.T, np.dot(W, Y))
        )
        P_est = np.dot(np.eye(self.n) - np.dot(gain, self.H), P_pred)

        return x_est, P_est, gain



class Algorithm_4(Algorithm):
    """
    Implementation of M Robust estimate filter with robust noise estimation
    """

    pass

