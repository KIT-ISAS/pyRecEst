import numpy as np

def generate_gaussian_like_grid_s3_v1(n_lat, n_lon):
    # Generate Gaussian-like latitudes
    beta = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    latitudes = np.arcsin(0.5 * (np.sin(beta[:-1]) + np.sin(beta[1:])))

    # Generate longitudes for each latitude
    longitudes = []
    for i, lat in enumerate(latitudes):
        n_lon_i = int(n_lon * np.cos(lat))
        longitudes.append(np.linspace(0, 2 * np.pi, n_lon_i, endpoint=False))

    # Generate Cartesian coordinates for the Gaussian-like grid on S^3
    grid_points = []
    for lat, lon_set in zip(latitudes, longitudes):
        for lon in lon_set:
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            w = np.sqrt(1 - x**2 - y**2 - z**2)
            grid_points.append([w, x, y, z])

    return np.array(grid_points)

n_lat = 10
n_lon = 20
grid_points_s3 = generate_gaussian_like_grid_s3_v1(n_lat, n_lon)

def generate_gaussian_grid_on_S3_v2(n_theta, n_phi, sigma_theta, sigma_phi):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    theta_weights = np.exp(-0.5 * (theta - np.pi/2)**2 / sigma_theta**2)
    theta_weights /= np.sum(theta_weights)
    phi_weights = np.exp(-0.5 * phi**2 / sigma_phi**2)
    phi_weights /= np.sum(phi_weights)

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    weights = np.outer(theta_weights, phi_weights)
    grid_points = np.stack([x, y, z], axis=-1)

    return grid_points, weights

n_theta = 10
n_phi = 10
sigma_theta = np.pi / 8
sigma_phi = np.pi / 8

grid_points, weights = generate_gaussian_grid_on_S3_v2(n_theta, n_phi, sigma_theta, sigma_phi)
print(grid_points)
