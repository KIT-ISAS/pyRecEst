from pyrecest.distributions import CustomLinearDistribution
import numpy as np

from scipy.stats import multivariate_normal as mvn
import scipy

def my_pdf(m,v,sigma,N):
    if m < 0:
        pdf = 0.0
    else:
        term1 = (m**N) / ((sigma**2) * v**(N-1))
        term2 = np.exp(-(m**2 + v**2)/(2*sigma**2))
        z = (m * v) / sigma**2
        term3 = scipy.special.iv(N-1,z)
        pdf = term1 * term2 * term3
    return np.squeeze(pdf)


my_pdf_fix = lambda xs: my_pdf(xs, v=1, sigma=1, N=1)

dist = CustomLinearDistribution(my_pdf_fix, dim=1, scale_by=1.0, shift_by=0.0)
samples = dist.sample_metropolis_hastings(100, start_point=np.array(0.5))


print(samples)

print(np.std(samples))
print(np.mean(samples))
