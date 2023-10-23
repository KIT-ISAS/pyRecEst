import copy
import warnings
from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import mod
from pyrecest.distributions import VonMisesDistribution

from .abstract_circular_filter import AbstractCircularFilter


class VonMisesFilter(AbstractCircularFilter):
    """
    A filter based on the Von Mises distribution.

    References:
    - M. Azmani, S. Reboul, J.-B. Choquel, and M. Benjelloun, "A recursive
      fusion filter for angular data" in 2009 IEEE International Conference
      on Robotics and Biomimetics (ROBIO), Dec. 2009, pp. 882-887.
    - Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
      Recursive Bayesian Filtering in Circular State Spaces
      arXiv preprint: Systems and Control (cs.SY), January 2015.
    """

    def __init__(self):
        """
        Constructor
        """
        AbstractCircularFilter.__init__(self, VonMisesDistribution(0, 1))

    def set_state(self, new_state: VonMisesDistribution):
        """
        Sets the current system state

        Parameters:
        new_state (VonMisesDistribution) : new state
        """
        self.filter_state = copy.deepcopy(new_state)

    def predict_identity(self, vmSys: VonMisesDistribution):
        """
        Predicts assuming identity system model, i.e.,
        x(k+1) = x(k) + w(k) mod 2*pi,
        where w(k) is additive noise given by vmSys.

        Parameters:
        vmSys (VMDistribution) : distribution of additive noise
        """
        self.filter_state = self.filter_state.convolve(vmSys)

    def update_identity(self, vmMeas: VonMisesDistribution, z=0.0):
        """
        Updates assuming identity measurement model, i.e.,
        z(k) = x(k) + v(k) mod 2*pi,
        where v(k) is additive noise given by vmMeas.

        Parameters:
        vmMeas (VMDistribution) : distribution of additive noise
        z : measurement in [0, 2pi)
        """
        assert z.shape in ((), (1,)), "z must be a scalar"
        if vmMeas.mu != 0.0:
            warning_message = (
                "The measurement noise is not centered at 0.0. "
                "This may lead to unexpected results because the "
                "update is based on the difference between the "
                "measurement and the current state."
            )
            warnings.warn(warning_message)

        muWnew = mod(z - vmMeas.mu, 2.0 * pi)
        vmMeasShifted = VonMisesDistribution(muWnew, vmMeas.kappa)
        self.filter_state = self.filter_state.multiply(vmMeasShifted)
