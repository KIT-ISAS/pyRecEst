from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from ..abstract_custom_distribution import AbstractCustomDistribution


class CustomHypersphericalDistribution(AbstractCustomDistribution, AbstractHypersphericalDistribution):

    def pdf(self, xs):
        assert xs.shape[-1] == self.input_dim, "Input dimension of pdf is not as expected"
        p = self.scale_by * self.f(xs)
        assert p.ndim <= 1,  "Output format of pdf is not as expected"
        return p

    @staticmethod
    def from_distribution(distribution):
        assert isinstance(distribution, AbstractHypersphericalDistribution)

        chd = CustomHypersphericalDistribution(distribution.pdf, distribution.dim)
        return chd
