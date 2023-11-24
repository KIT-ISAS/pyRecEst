import numpy as np

class RadialMeasurementFunction:
    def __init__(self, measurement, nr_Fourier_coeff):
        self.measurement = measurement
        self.nr_Fourier_coeff = nr_Fourier_coeff
    
    def calc_fourier_coeff(self, theta):
        fourier_coeff = zeros((self.nr_Fourier_coeff, len(theta)))
        fourier_coeff[0, :] = 0.5
        index = 1
        for i in range(1, self.nr_Fourier_coeff, 2):
            fourier_coeff[i:i+2, :] = [np.cos(index * theta), np.sin(index * theta)]
            index += 1
        return fourier_coeff
    
    def measurement_equation(self, state_samples, noise_samples):
        y = self.measurement
        x = state_samples
        numberOfSigmaPoints = state_samples.shape[1]
        pseudo_measurement = zeros((2, numberOfSigmaPoints))
        
        s = noise_samples[0, :]
        v = noise_samples[1:3, :]
        b = x[:self.nr_Fourier_coeff, :]
        m = x[self.nr_Fourier_coeff:self.nr_Fourier_coeff + 2, :]
        
        theta = np.arctan2(y[1] - m[1, :], y[0] - m[0, :]) + 2 * np.pi
        R = self.calc_fourier_coeff(theta)
        e = np.array([np.cos(theta), np.sin(theta)])
        
        for i in range(len(s)):
            pseudo_measurement[:, i] = s[i] * R[:, i].T @ b[:, i] * e[:, i] + m[:, i] + v[:, i]
        
        return pseudo_measurement

def main():
    measurement_space_dim = 3
    object_space_dim = 5
    
    # Generate a random hypersurface model
    A, b = generate_random_hypersurface(measurement_space_dim, object_space_dim)
    
    # Define an example object state
    object_state = np.random.rand(object_space_dim, 1)
    
    # Map the object state to a measurement using the random hypersurface model
    measurement = map_object_to_measurement(A, object_state, b)
    
    # Define the number of Fourier coefficients
    nr_Fourier_coeff = 4
    
    # Create a RadialMeasurementFunction object
    radial_measurement_function = RadialMeasurementFunction(measurement, nr_Fourier_coeff)
    
    # Define some example state samples and noise samples
    state_samples = np.random.rand(object_space_dim, 3)
    noise_samples = np.random.rand(3, 3)
    
    # Calculate the pseudo-measurements using the RadialMeasurementFunction object
    pseudo_measurements = radial_measurement_function.measurement_equation(state_samples, noise_samples)
    
    print("Object state:")
    print(object_state)
    
    print("\nMeasurement:")
    print(measurement)
    
    print("\nPseudo-measurements:")
    print(pseudo_measurements)

if __name__ == "__main__":
    main()
