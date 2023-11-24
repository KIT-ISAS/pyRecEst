# Based on Matlab code by UDH

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

wxy = None
SX = None
SY = None

def glcd(x, y, wxy_, SX_, SY_):
    assert x.ndim == 1, 'x must be column vector'
    assert y.ndim == 1, 'y must be column vector'
#    assert wxy_.ndim == 1, 'wxy_ must be column vector'

    wxy = wxy_
    SX = SX_
    SY = SY_
    xb = marshal(x, y)
    objective_function(xb)
    gradient_function(xb)
    res = minimize(objective_function, xb, method='L-BFGS-B', jac=gradient_function, options={'disp': True, 'ftol': 1e-22, 'maxiter': 5000})
    x, y = unmarshal(res.x)
    return x, y, res.fun

def objective_function(xb):
    x, y = unmarshal(xb)
    lambda_ = np.array([1000, 1000, 10, 10, 10])
    f = distance_measure_gaussian_numeric(wxy, x, y, SX, SY, lambda_)
    return f

def gradient_function(xb):
    x, y = unmarshal(xb)
    lambda_ = np.array([1000, 1000, 10, 10, 10])
    r = gradient_gaussian_numeric(wxy, x, y, SX, SY, lambda_)
    return r

def distance_measure_gaussian_numeric(wxy, x, y, SX, SY, lambda_):
    flag = 0
    sx = SX
    sy = SY
    bmax = 20
    Cb = np.log(4 * bmax ** 2) - 0.577216
    b = np.linspace(0.0001, np.sqrt(bmax), 100) ** 2
    xx = np.column_stack((x, y))
    N = 2
    L = wxy.shape[0]

    if flag == 0:
        flag = 1
        G1 = np.pi ** (N / 2) * b ** (N + 1) / (np.sqrt(sx ** 2 + b ** 2) * np.sqrt(sy ** 2 + b ** 2))
        pp = UnivariateSpline(b, G1, s=0)
        
        """
        # Plotting (begin)
        cs = CubicSpline(b, G1)

        # Evaluate the spline at finer points
        b_fine = np.linspace(np.min(b), np.max(b), 100)
        G1_fine = cs(b_fine)

        # Plot the original data and the spline
        plt.plot(b, G1, 'bo', markersize=10, label='Data')  # original data
        plt.plot(b_fine, G1_fine, 'r-', label='Spline')  # spline
        plt.xlabel('b')
        plt.ylabel('G1')
        plt.title('Spline Interpolation')
        plt.legend()
        plt.show()
        """
        
        # Plotting (end)
        
        G1, _ = quad(pp, 0, bmax)

    G2t = 0

    for i in range(L):
        G2t += wxy[i] * np.exp(-0.5 * (xx[i, 0] ** 2 / (sx ** 2 + 2 * b ** 2) + xx[i, 1] ** 2 / (sy ** 2 + 2 * b ** 2)))

    G2 = -2 * (2 * np.pi) ** (N / 2) * b ** (N + 1) / (np.sqrt(sx ** 2 + 2 * b ** 2) * np.sqrt(sy ** 2 + 2 * b ** 2)) * G2t
    pp = UnivariateSpline(b, G2, s=0)
    G2, _ = quad(pp, 0, bmax)

    Mxx = np.subtract.outer(x, x).T
    Myy = np.subtract.outer(y, y).T
    T = Mxx ** 2 + Myy ** 2
    G3 = np.squeeze(np.pi * wxy.T @ (4 * bmax ** 2 * np.exp(-0.5 * T / (2 * bmax ** 2)) - Cb * T + xlog(T) - T ** 2 / (4 * bmax ** 2)) @ wxy / 8)

    G = G1 + G2 + G3 + lambda_[0] * (wxy.T @ x) ** 2 + lambda_[1] * (wxy.T @ y) ** 2 + \
        lambda_[2] * (wxy.T @ (x ** 2) - sx ** 2) ** 2 + lambda_[3] * (wxy.T @ (x * y)) ** 2 + \
        lambda_[4] * (wxy.T @ (y ** 2) - sy ** 2) ** 2

    return G

def gradient_gaussian_numeric(wxy, x, y, SX, SY, lambda_):
    s = np.array([SX, SY])
    bmax = 20
    Cb = np.log(4 * bmax ** 2) - 0.577216
    xx = np.column_stack((x, y))
    N = 2
    L = len(wxy)
    db = 0.005
    b = np.arange(db, bmax + db, db)

    H = 2 * (2 * np.pi) ** (N / 2) * b ** (N + 1) / (np.sqrt(s[0] ** 2 + 2 * b ** 2) * np.sqrt(s[1] ** 2 + 2 * b ** 2))

    G1 = zeros((2 * L, len(b)))

    for eta in range(2):
        k = H / (s[eta] ** 2 + 2 * b ** 2)
        for i in range(L):
            G1[eta * L + i, :] = wxy[i] * xx[i, eta] * k * np.exp(-0.5 * (xx[i, 0] ** 2 / (s[0] ** 2 + 2 * b ** 2) + xx[i, 1] ** 2 / (s[1] ** 2 + 2 * b ** 2)))

    G1 = db * np.sum(G1, axis=1)

    Mxx = np.subtract.outer(x, x).T
    Myy = np.subtract.outer(y, y).T
    M = Mxx ** 2 + Myy ** 2
    T = plog(M) - M / (4 * bmax ** 2)

    rx = (wxy.T @ (Mxx * T)).T
    ry = (wxy.T @ (Myy * T)).T

    G2 = np.pi * np.hstack((np.squeeze(rx) + Cb * (wxy.T @ x - x), np.squeeze(ry) + Cb * (wxy.T @ y - y))) / (2 * L)

    G3 = np.hstack((
        np.squeeze(2 * lambda_[0] * (wxy.T @ x) * wxy) + 4 * (wxy.T @ (x ** 2) - s[0] ** 2) * lambda_[2] * np.squeeze(wxy) * x + 2 * (wxy.T @ (x * y)) * lambda_[3] * np.squeeze(wxy) * y,
        np.squeeze(2 * lambda_[1] * (wxy.T @ y) * wxy) + 4 * (wxy.T @ (y ** 2) - s[1] ** 2) * lambda_[4] * np.squeeze(wxy) * y + 2 * (wxy.T @ (x * y)) * lambda_[3] * np.squeeze(wxy) * x))

    G = G1 + G2 + G3

    return G

def marshal(x, y):
    return np.concatenate((x, y))

def unmarshal(xb):
    L = len(xb) // 2
    x = xb[:L]
    y = xb[L:]
    return x, y

def plog(x):
    x = np.array(x)
    indx = (x == 0)
    x[indx] = 1
    y = np.log(x)
    return y

def xlog(x):
    return x * plog(x)

def randn_box_muller(n_samples):
    def box_muller_transform(u1, u2):
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        return z1, z2

    uniform_random_numbers = np.random.rand(n_samples)
    # Generate normally distributed random numbers using the Box-Muller transform
    normal_random_numbers = []
    for i in range(0, len(uniform_random_numbers), 2):
        z1, z2 = box_muller_transform(uniform_random_numbers[i], uniform_random_numbers[i+1])
        normal_random_numbers.append(z1)
        normal_random_numbers.append(z2)

    print("Random numbers in Python:", normal_random_numbers[:5])
    return normal_random_numbers


if __name__ == "__main__":
    SX = 1
    SY = 0.7
    L = 10

    for SY in np.arange(0.9, 0.001, -0.02):
        wxy = np.ones((L,1)) / L
        seed_value = 42
        np.random.seed(seed_value)
        

        x = SX * np.array(randn_box_muller(int(L)))
        y = SY * np.array(randn_box_muller(int(L)))

        x, y, G = glcd(x, y, wxy, SX, SY)
        plt.cla()
        plt.plot(x, y, '.', markeredgecolor=[1, 1, 1], markersize=10)
        plt.plot(x, y, 'r.', markersize=7)
        plt.axis('equal')
        plt.gca().set_xlim([-4, 4])
        plt.gca().set_ylim([-4, 4])
        plt.draw()
        plt.show(block=False)
        plt.pause(.001)
