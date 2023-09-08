import numpy as np
import matplotlib.pyplot as plt


def plot_ellipsoid(center, shape_matrix, scaling_factor=1, color="blue"):
    if center.shape[0] == 2:
        plot_ellipsoid_2d(center, shape_matrix, scaling_factor, color)
    elif center.shape[0] == 3:
        plot_ellipsoid_3d(center, shape_matrix, scaling_factor, color)
    else:
        raise ValueError("Only 2D and 3D ellipsoids are supported.")


def plot_ellipsoid_2d(center, shape_matrix, scaling_factor=1, color="blue"):
    xs = np.linspace(0, 2 * np.pi, 100)
    ps = scaling_factor * shape_matrix @ np.column_stack((np.cos(xs), np.sin(xs)))
    plt.plot(ps[0] + center[0], ps[1] + center[1], color=color)
    plt.show()


def plot_ellipsoid_3d(center, shape_matrix, scaling_factor=1, color="blue"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    V, D = np.linalg.eig(shape_matrix)
    all_coords = V @ np.sqrt(D) @ np.array([x.ravel(), y.ravel(), z.ravel()]) + center.reshape(-1, 1)
    x = np.reshape(all_coords[0], x.shape)
    y = np.reshape(all_coords[1], y.shape)
    z = np.reshape(all_coords[2], z.shape)

    ax.plot_surface(
        scaling_factor * x, scaling_factor * y, scaling_factor * z, color=color, alpha=0.7, linewidth=0, antialiased=False
    )
    plt.show()