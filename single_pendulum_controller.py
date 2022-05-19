import control
import numpy as np
from numpy import cos, sin

class SinglePendulumController():
    """
    Initializes a single inverted pendulum controller object

    Inputs::

    mass_cart --
        Defines the mass of the cart (kg)

    mass_pendulum --
        Defines the mass of the pendulum (kg)

    length_pendulum --
        Defines the length of the pendulum (m)

    friction_coefficient --
        Defines the friction coefficient between the cart and the ground
    """
    def __init__(self, mass_cart, mass_pendulum, length_pendulum, friction_coefficient):
        self._name = "single-inverted-pendulum-controller"
        self._states = [None, None, None, None]
        self._output = 0
        self._time = 0
        self._mc = mass_cart
        self._mp = mass_pendulum
        self._l = length_pendulum / 2
        self._b = friction_coefficient
        self._I = (self._mp * (2 * self._l)**2) / 12
        self._g = 9.81
        self._d0 = self._mc + self._mp
        self._d1 = self._mp * self._l ** 2 + self._I
        self._d2 = self._mp * self._l
        self._parameters = {"mass_cart" : self._mc, "mass_pendulum" : self._mp, 
                            "length_pendulum" : 2 * self._l, "inertia_pendulum" : self._I,
                            "friction_coefficient" : self._b}
        self._det = self._d1 * self._d0 - self._d2**2
        self._A = np.array([[0, 1, 0, 0],
                            [0, - self._d1 * self._b / self._det, - self._d2**2 * self._g, 0],
                            [0, 0, 0, 1],
                            [0, self._d2 * self._b, self._d0 * self._d2 * self._g, 0]])
        self._B = np.array([0, self._d1 / self._det, 0, - self._d2 / self._det]).reshape(-1, 1)
        self._C = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]])
        self._D = np.zeros([2, 1])
        if not self._is_observable():
            raise Exception("The system is not observable with the given measurments and canot be estimated.")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, latest_time):
        self._time = latest_time

    def get(self, string):
        """Returns the value of the specified parameter via string if it exists else 0"""
        for key in self._parameters:
            if key == string:
                return self._parameters[string]
        print("Warning: Could not find the specified parameter.")
        return 0

    def _is_observable(self):
        """
        Checks whether the system is observable given matrices A, C
        
        Returns::
            True: if the system is observable, else False
        """
        if np.linalg.matrix_rank(control.obsv(self._A, self._C)) == len(self._A):
            return True
        else:
            return False