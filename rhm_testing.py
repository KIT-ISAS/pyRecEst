import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

"""
def random_hypersurface_model(number_of_measurement):
    # Number of Fourier coefficients
    nr_Fourier_coeff = 11

    # State description prior [b0--bn, x, y]
    x = zeros((nr_Fourier_coeff + 2, 1))
    x[0] = 1.5

    # State covariance prior
    C_x = np.diag(np.concatenate((np.ones(nr_Fourier_coeff) * 0.02, np.array([0.3, 0.3]))))

    # Measurement noise
    measurement_noise = np.diag(np.array([0.2, 0.2])**2)

    # Scale properties
    scale_mean = 0.7
    scale_variance = 0.08

    # Shape resolution for plotting
    phi_vec = np.arange(0, 2 * np.pi, 0.01)

    # Object size
    a = 3
    b = 0.5
    c = 2
    d = 0.5

    size_object = [a, b, c, d]

    # Object shape bounds
    object_bounds = np.array([[-d, -c], [d, -c], [d, -b], [a, -b], [a, b], [d, b], [d, c],
                              [-d, c], [-d, b], [-a, b], [-a, -b], [-d, -b]]).T / 2

    # Main

    # Plot
    fig, ax = plt.subplots()
    h_object = ax.fill(object_bounds[0], object_bounds[1], color=[0.7, 0.7, 0.7])
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('x-Axis')
    ax.set_ylabel('y-Axis')
    ax.set_title('Random Hypersurface Model Siμlation')

    for j in range(number_of_measurement):
        # Get new measurement
        new_measurement = get_new_measurement(size_object, measurement_noise)

        # Filter step
        x, C_x = UKF_FilterStep(x, C_x, new_measurement, np.array([[scale_mean], [0], [0]]),
                                np.diag([scale_variance] + list(np.diag(measurement_noise))), nr_Fourier_coeff)

        # Plot
        shape = calc_shape(phi_vec, x, nr_Fourier_coeff)

        h_measure = ax.plot(new_measurement[0], new_measurement[1], '+')
        h_shape = ax.plot(shape[0], shape[1], 'g-', linewidth=2)
        ax.legend(['Target', 'Measurement', 'Estimated shape'])
        plt.draw()
        plt.pause(0.1)

        if j != number_of_measurement - 1:
            h_shape[0].remove()

    plt.show()
"""


def random_hypersurface_model(measurements):
    # Number of Fourier coefficients
    nr_Fourier_coeff = 11

    # State description prior [b0--bn, x, y]
    x = zeros((nr_Fourier_coeff + 2, 1))
    x[0] = 1.5

    # State covariance prior
    C_x = np.diag(np.concatenate((np.ones(nr_Fourier_coeff) * 0.02, np.array([0.3, 0.3]))))

    # Measurement noise
    measurement_noise = np.diag(np.array([0.2, 0.2])**2)

    # Scale properties
    scale_mean = 0.7
    scale_variance = 0.08

    # Shape resolution for plotting
    phi_vec = np.arange(0, 2 * np.pi, 0.01)

    # Main

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal')
    ax.set_xlabel('x-Axis')
    ax.set_ylabel('y-Axis')
    ax.set_title('Random Hypersurface Model Siμlation')

    for new_measurement in measurements:
        # Filter step
        #x, C_x = UKF_FilterStep(x, C_x, new_measurement, np.array([[scale_mean], [0], [0]]),
        #                        np.diag([scale_variance] + list(np.diag(measurement_noise))), nr_Fourier_coeff)
        x, C_x = UKF_FilterStep(x, C_x, new_measurement, np.array([[scale_mean], [0], [0]]),
                        blkdiag(scale_variance, measurement_noise),
                        measurement_function, nr_Fourier_coeff)


        # Plot
        shape = calc_shape(phi_vec, x, nr_Fourier_coeff)

        h_measure = ax.plot(new_measurement[0], new_measurement[1], '+')
        h_shape = ax.plot(shape[0], shape[1], 'g-', linewidth=2)
        ax.legend(['Target', 'Measurement', 'Estimated shape'])
        plt.draw()
        plt.pause(0.1)

        if new_measurement != measurements[-1]:
            h_shape[0].remove()

    plt.show()

    return shape

def get_new_measurement(size_object, measurement_noise):
    """
    Generates a new measurement for a given size object and measurement noise.

    :param size_object: numpy array of the size object, expected shape (4,)
    :param measurement_noise: numpy array of the measurement noise, expected shape (2,)
    :return: numpy array of the new measurement, expected shape (2, 1)
    """
    a, b, c, d = size_object

    # Generate random angle
    theta = np.random.uniform(0, 2 * np.pi)

    # Compute x and y coordinates for the given angle and size object parameters
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    # Add measurement noise
    measurement = np.array([[x], [y]]) + np.random.μltivariate_normal([0, 0], np.diag(measurement_noise)).reshape(2, 1)

    return measurement

"""
def get_new_measurement(size_object, measurement_noise):
    a = size_object[0]  # -- width of the horizontal rectangle
    b = size_object[1]  # | height of the horizontal rectangle
    c = size_object[2]  # | height of the vertical rectangle
    d = size_object[3]  # -- width of the vertical rectangle

    measurement_source_not_valid = True

    while measurement_source_not_valid:
        # Measurement source
        x = -a / 2 + a * np.random.rand()
        y = -c / 2 + c * np.random.rand()

        if (y > b / 2 and x < -d / 2) or (y > b / 2 and x > d / 2) or \
                (y < -b / 2 and x < -d / 2) or (y < -b / 2 and x > d / 2):
            x = -a / 2 + a * np.random.rand()
            y = -c / 2 + c * np.random.rand()
        else:
            measurement_source_not_valid = False

    # Add zero-mean Gaussian noise to the measurement sources
    measurement = np.array([[x], [y]]) + np.random.μltivariate_normal([0, 0], measurement_noise).reshape(2, 1)

    return measurement
"""

def f_meas_pseudo_squared(x, noise, y, nr_Fourier_coeff):
    number_of_sigma_points = x.shape[1]
    pseudo_measurement = zeros((1, number_of_sigma_points))

    for j in range(number_of_sigma_points):
        s = noise[0, j]
        v = noise[1:3, j]
        b = x[:nr_Fourier_coeff, j]
        m = x[nr_Fourier_coeff:nr_Fourier_coeff + 2, j]

        theta = np.arctan2(y[1] - m[1], y[0] - m[0]) + 2 * np.pi

        R = calc_fourier_coeff(theta, nr_Fourier_coeff)

        e = np.array([np.cos(theta), np.sin(theta)])

        pseudo_measurement[0, j] = (np.linalg.norm(m - y))**2 - (s**2 * (R @ b)**2 + 2 * s * R @ b * e.T @ v + np.linalg.norm(v)**2)

    return pseudo_measurement


def calc_fourier_coeff(theta, nr_Fourier_coeff):
    fourier_coeff = zeros(nr_Fourier_coeff)
    fourier_coeff[0] = 0.5

    index = 1
    for i in range(1, nr_Fourier_coeff, 2):
        fourier_coeff[i:i + 2] = [np.cos(index * theta), np.sin(index * theta)]
        index += 1

    return fourier_coeff


def calc_shape(phi_vec, x, nr_Fourier_coeff):
    shape = zeros((2, len(phi_vec)))

    for i in range(len(phi_vec)):
        phi = phi_vec[i]
        R = calc_fourier_coeff(phi, nr_Fourier_coeff)
        e = np.array([np.cos(phi), np.sin(phi)]).reshape(2, 1)
        shape[:, i] = (R @ x[:-2] * e + x[-2:]).flatten()

    return shape
 

# ... (previous code)
def UKF_FilterStep(x, C, measurement, measurement_noise_mean, measurement_noise_covariance, measurement_function_handle, numberOfFourierCoef):

    def transition_function(state, dt):
        return state

    def measurement_function(state, noise, measurement_noise_mean):
        b = state[:nr_Fourier_coeff]
        m = state[nr_Fourier_coeff:nr_Fourier_coeff + 2]

        theta = np.arctan2(measurement[1] - m[1], measurement[0] - m[0]) + 2 * np.pi

        R = calc_fourier_coeff(theta, nr_Fourier_coeff)
        s = noise[0]
        v = noise[1:3]

        e = np.array([np.cos(theta), np.sin(theta)])

        return (np.linalg.norm(m - measurement))**2 - (s**2 * (R @ b)**2 + 2 * s * R @ b * e.T @ v + np.linalg.norm(v)**2)

    n = len(x)
    alpha = 1
    beta = 0
    kappa = 0

    merwe_points = MerweScaledSigmaPoints(n, alpha=alpha, beta=beta, kappa=kappa)
    ukf = UnscentedKalmanFilter(n, 1, dt=0, fx=transition_function, hx=measurement_function, points=merwe_points)
    ukf.x = x.flatten()
    ukf.P = C
    ukf.Q = measurement_noise_covariance
    ukf.R = np.array([[0.01]])
    ukf.hx_args = (measurement_noise_mean,)  # Pass additional arguments using hx_args attribute

    ukf.update(measurement.flatten(), R=np.array([[0.01]]))

    x_e = ukf.x.reshape(-1, 1)
    C_e = ukf.P

    return x_e, C_e

"""
def UKF_FilterStep(x, C, measurement, measurement_noise_mean, measurement_noise_covariance, nr_Fourier_coeff):
    def transition_function(state, dt):
        return state

    def measurement_function(state, noise):
        b = state[:nr_Fourier_coeff]
        m = state[nr_Fourier_coeff:nr_Fourier_coeff + 2]

        theta = np.arctan2(measurement[1] - m[1], measurement[0] - m[0]) + 2 * np.pi

        R = calc_fourier_coeff(theta, nr_Fourier_coeff)
        s = noise[0]
        v = noise[1:3]

        e = np.array([np.cos(theta), np.sin(theta)])

        return (np.linalg.norm(m - measurement))**2 - (s**2 * (R @ b)**2 + 2 * s * R @ b * e.T @ v + np.linalg.norm(v)**2)

    n = len(x)
    alpha = 1
    beta = 0
    kappa = 0

    merwe_points = MerweScaledSigmaPoints(n, alpha=alpha, beta=beta, kappa=kappa)
    ukf = UnscentedKalmanFilter(n, 1, dt=0, fx=transition_function, hx=measurement_function, points=merwe_points)
    ukf.x = x.flatten()
    ukf.P = C
    ukf.Q = measurement_noise_covariance
    ukf.R = np.array([[0.01]])

    ukf.update(measurement.flatten(), R=np.array([[0.01]]), hx_args=(measurement_noise_mean,))

    x_e = ukf.x.reshape(-1, 1)
    C_e = ukf.P

    return x_e, C_e
"""
# ... (rest of the code)


def test_random_hypersurface_model():
    np.random.seed(42)
    
    # Test parameters
    number_of_measurements = 50
    true_size_object = np.array([3, 0.5, 2, 0.5])
    measurement_noise_std = np.array([0.2, 0.2])
    
    # Generate true and noisy measurements
    true_measurements = [get_new_measurement(true_size_object, zeros(2)) for _ in range(number_of_measurements)]
    noisy_measurements = [m + np.random.randn(2) * measurement_noise_std for m in true_measurements]
    
    # Run the random hypersurface model with the noisy measurements
    final_shape = random_hypersurface_model(noisy_measurements)
    
    # Compute the true shape from the true measurements
    true_shape = calc_true_shape(true_measurements, true_size_object)
    
    # Compare the estimated shape with the true shape
    error = np.mean(np.abs(final_shape - true_shape))
    print(f"Mean absolute error between estimated shape and true shape: {error}")
    assert error < 0.5, "Error between estimated shape and true shape is too large."

def calc_true_shape(true_measurements, true_size_object):
    phi_vec = np.arange(0, 2 * np.pi, 0.01)
    
    # Compute the true shape using the average of true measurements
    true_center = np.mean(true_measurements, axis=0)
    true_radius = np.mean([np.linalg.norm(m - true_center) for m in true_measurements])
    true_shape = np.array([true_center + true_radius * np.array([np.cos(phi), np.sin(phi)]) for phi in phi_vec]).T

    return true_shape

if __name__ == "__main__":
    test_random_hypersurface_model()