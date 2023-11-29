import numpy as np
from lcd_computation import Computation
import matplotlib.pyplot as plt

def main():
    computation = Computation()
    dimension = 2
    # num_samples = 21
    num_samples = 20
    initial_parameters = np.array([])

    samples, result = computation(dimension, num_samples, initial_parameters)

    print(">> Samples approximating a three-dimensional standard normal distribution:")
    print(samples, end='\n\n')

    mean = np.mean(samples, axis=0)

    print(">> Sample mean:")
    print(mean, end='\n\n')

    diffs = samples - np.expand_dims(mean, 0)

    print(">> Sample covariance matrix:")
    print((diffs.T @ diffs) / (num_samples - 1), end='\n\n')

    print(">> The optimization was initialized with these parameters:")
    print(initial_parameters, end='\n\n')

    print(">> Information about the optimization:\n")
    print(result)
    
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()

if __name__ == '__main__':
    main()
