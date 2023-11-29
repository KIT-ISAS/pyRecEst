import numpy as np
from lie_learn.representations.SO3.wigner_d import wigner_d_matrix

# Define the Euler angles (in radians)
alpha = np.pi / 3
beta = np.pi / 4
gamma = np.pi / 6

# Set the angular momentum quantum number (j)
j = 1

# Compute the Wigner D-matrix elements
D_matrix = wigner_d_matrix(j, alpha, beta, gamma)

print(f"Wigner D-matrix elements for j = {j} and Euler angles (alpha, beta, gamma) = ({alpha}, {beta}, {gamma}):")
print(D_matrix)