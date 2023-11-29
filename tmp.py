import numpy as np

def box_muller_transform(u1, u2):
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2

seed_value = 42
np.random.seed(seed_value)
uniform_random_numbers = np.random.rand(6)

# Generate normally distributed random numbers using the Box-Muller transform
normal_random_numbers = []
for i in range(0, len(uniform_random_numbers), 2):
    z1, z2 = box_muller_transform(uniform_random_numbers[i], uniform_random_numbers[i+1])
    normal_random_numbers.append(z1)
    normal_random_numbers.append(z2)

print("Random numbers in Python:", normal_random_numbers[:5])
