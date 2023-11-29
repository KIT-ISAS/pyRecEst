import numpy as np

def mean_correction(samples):
    num_samples = samples.shape[1]
    weight = 1.0 / num_samples
    mean = weight * np.sum(samples, axis=1)
    corrected_samples = samples - mean[:, np.newaxis]
    return corrected_samples

def covariance_correction(samples):
    num_samples = samples.shape[1]
    weight = 1.0 / num_samples
    cov = weight * samples.T @ samples
    corrected_samples = (np.linalg.cholesky(np.linalg.inv(cov)).T @ samples.T).T
    return corrected_samples

def std_normal_rnd_matrix(rows, cols):
    rnd_matrix = np.random.normal(0.0, 1, (rows, cols))
    return rnd_matrix

def normalized_cov_error(samples):
    num_samples = samples.shape[1]
    dim = samples.shape[0]
    cov = samples @ samples.T / num_samples
    cov_error = np.linalg.norm(cov - np.identity(dim)) / (dim * dim)
    return cov_error
