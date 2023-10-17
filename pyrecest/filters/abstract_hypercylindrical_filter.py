from .abstract_lin_periodic_filter import AbstractLinPeriodicFilter


class AbstractHypercylindricalFilter(AbstractLinPeriodicFilter):
    def get_point_estimate(self):
        return self.filter_state.hybrid_mean()