import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from gaussian_distribution import GaussianDistribution

def randomHypersurfaceModelExample(numberOfMeasurement):
    nr_Fourier_coeff = 11

    x = zeros(nr_Fourier_coeff + 4) # 2 for pos, 2 for vel
    x[0] = 1.5

    C_x = np.diag(np.concatenate([np.ones(nr_Fourier_coeff) * 0.02, [0.3, 0.3, 0.3, 0.3]]))

    measurementNoise = np.diag([0.2, 0.2])**2

    scale_mean = 0.7
    scale_variance = 0.08

    phi_vec = np.linspace(0, 2 * np.pi, 629)

    a = 3
    b = 0.5
    c = 2
    d = 0.5

    sizeObject = [a, b, c, d]

    objectBounds = np.array([[-d, -c], [d, -c], [d, -b], [a, -b], [a, b], [d, b], [d, c],
                    [-d, c], [-d, b], [-a, b], [-a, -b], [-d, -b]]) / 2

    fig, ax = plt.subplots()
    ax.fill(objectBounds[:, 0], objectBounds[:, 1], [.7, .7, .7])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x-Axis')
    ax.set_ylabel('y-Axis')
    ax.set_title('Random Hypersurface Model Siμlation')

    for j in range(numberOfMeasurement):
        if j != 0:
            x, C_x = UKF_PredictionStep(x, C_x)
        
        newMeasurement = getNewMeasurement(sizeObject, measurementNoise)
        x, C_x = UKF_FilterStep(x, C_x, newMeasurement, np.array([scale_mean, 0, 0]),
                                #x_ukf = np.concatenate((x, measurementNoiseMean)),
                                #np.block([[scale_variance], [measurementNoise]]),
                                np.block([
                                    [scale_variance, zeros((1, measurementNoise.shape[1]))],
                                    [zeros((measurementNoise.shape[0], 1)), measurementNoise]
                                ]),
                                f_meas_pseudo_squared, nr_Fourier_coeff)
        # def UKF_FilterStep(x, C, measurement, measurementNoiseMean, measurementNoiseCovariance, measurementFunctionHandle, numberOfFourierCoef):

        shape = calcShape(phi_vec, x, nr_Fourier_coeff)

        h_measure = ax.plot(newMeasurement[0], newMeasurement[1], '+')
        h_shape = ax.plot(shape[0, :], shape[1, :], 'g-', linewidth=2)
        ax.legend(['Target', 'Measurement', 'Estimated shape'])

        plt.draw()
        plt.pause(0.001)

        if j != numberOfMeasurement - 1:
            h_shape[0].remove()

    plt.show()

def getNewMeasurement(sizeObject, measurementNoise):
    # For testing
    #"""
    a, b, c, d = sizeObject

    measurementsourceNotValid = True

    while measurementsourceNotValid:
        x = -a/2 + a * np.random.rand()
        y = -c/2 + c * np.random.rand()

        if (y > b/2 and x < -d/2) or (y > b/2 and x > d/2) or \
           (y < -b/2 and x < -d/2) or (y < -b/2 and x > d/2):
            x = -a/2 + a * np.random.rand()
            y = -c/2 + c * np.random.rand()
        else:
            measurementsourceNotValid = False

    measurement = np.array([x, y]) + np.random.μltivariate_normal(zeros(2), measurementNoise)
    return measurement
    #"""
    #return np.array([1.0, 1.0])

def f_meas_pseudo_squared(x, noise, y, nr_Fourier_coeff):
    numberOfSigmaPoints = x.shape[1]
    pseudoMeasurement = zeros(numberOfSigmaPoints)

    for j in range(numberOfSigmaPoints):
        s = noise[0, j]
        v = noise[1:3, j]
        b = x[:nr_Fourier_coeff, j]
        m = x[nr_Fourier_coeff:nr_Fourier_coeff + 4, j]

        theta = np.arctan2(y[1] - m[1], y[0] - m[0]) + 2 * np.pi
        R = calcFourierCoeff(theta, nr_Fourier_coeff)

        e = np.array([np.cos(theta), np.sin(theta)])

        pseudoMeasurement[j] = (np.linalg.norm(m - y))**2 - (s**2 * (R @ b)**2 + 2 * s * R @ b * e @ v + np.linalg.norm(v)**2)

    return pseudoMeasurement

def calcFourierCoeff(theta, nr_Fourier_coeff):
    fourie_coff = zeros(nr_Fourier_coeff)
    fourie_coff[0] = 0.5

    index = 0
    for i in range(nr_Fourier_coeff, 2):
        fourie_coff[i:i + 2] = [np.cos(index * theta), np.sin(index * theta)]
        index += 1

    return fourie_coff

def calcShape(phi_vec, x, nr_Fourier_coeff):
    shape = zeros((2, len(phi_vec)))

    for i in range(len(phi_vec)):
        phi = phi_vec[i]
        R = calcFourierCoeff(phi, nr_Fourier_coeff)
        e = np.array([np.cos(phi), np.sin(phi)])
        shape[:, i] = R @ x[:-4] * e + x[-4:-2]

    return shape

def UKF_FilterStep(x, C, measurement, measurementNoiseMean, measurementNoiseCovariance, measurementFunctionHandle, numberOfFourierCoef):
    alpha = 1
    beta = 0
    kappa = 0

    x_ukf = np.concatenate([x, measurementNoiseMean])

    C_ukf = np.block([[C, zeros((C.shape[0], measurementNoiseCovariance.shape[1]))],
                      [zeros((measurementNoiseCovariance.shape[0], C.shape[1])), measurementNoiseCovariance]])

    C_ukf = nearestPD(C_ukf)

    n = x_ukf.size
    n_state = x.size

    lamda = alpha**2 * (n + kappa) - n

    WM = np.ones(2 * n + 1) / (2 * (n + lamda))
    WM[0] = lamda / (n + lamda)

    WC = np.ones(2 * n + 1) / (2 * (n + lamda))
    WC[0] = (lamda / (n + lamda)) + (1 - alpha**2 + beta)

    A = np.sqrt(n + lamda) * np.linalg.cholesky(C_ukf).T

    xSigma = np.hstack([zeros((x_ukf.size, 1)), -A, A])
    xSigma = xSigma + np.tile(x_ukf, (xSigma.shape[1], 1)).T

    z = zeros(1)
    C_yy = zeros((1, 1))
    C_xy = zeros((n_state, 1))
    #C_xy = zeros((1, 1))

    zSigmaPredict = measurementFunctionHandle(xSigma[:n_state, :], xSigma[n_state:n, :], measurement, numberOfFourierCoef)

    # z = np.sum(zSigmaPredict * WM, axis=1)
    z = np.sum(zSigmaPredict * WM, axis=0)

    for i in range(zSigmaPredict.shape[0]):
        C_yy += WC[i] * np.outer((zSigmaPredict[i] - z), (zSigmaPredict[i] - z))
        C_xy += WC[i] * np.outer((xSigma[:n_state, i] - x), (zSigmaPredict[i] - z))

    K = C_xy @ np.linalg.inv(C_yy)
    #x_e = x + K @ (zeros_like(z) - z)
    x_e = x + (K @ np.reshape((zeros_like(z) - z), (-1,1))).flatten()
    C_e = C - K @ C_yy @ K.T

    return x_e, C_e

# Main code to run the siμlation
if __name__ == "__main__":
    numberOfMeasurement = 10
    randomHypersurfaceModelExample(numberOfMeasurement)

