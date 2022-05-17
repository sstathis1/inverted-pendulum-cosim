from numpy import cos, sin
import numpy as np
from scipy.integrate import solve_ivp

class SinglePendulum():
    """
    Initializes a single inverted pendulum object

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
        self._name = "single-inverted-pendulum"
        self._states = [None, None]
        self._output = [None, None]
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
                            "length_pendulum" : 2 * self._l, "friction_coefficient" : self._b,
                            "inertia_pendulum" : self._I}

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

    def ode(self, t, x):
        """
        Contains the system of ordinary differential equations for the single-pendulum on a cart.

        Inputs::
        t --
            time (s)
        x --
            state [x (m), v (m/s), theta (rad), omega (rad/s)]
        """
        delta = (self._d0) * (self._d1) - (self._d2 * cos(x[2]))**2
        return  [x[1], - 1 / delta * (self._d1 * (self._d2 * sin(x[2]) * x[3]**2 + self.input(t))
                 - (self._d2 * cos(x[2])) * (self._d2 * self._g * sin(x[2]))), x[3], 
                 - 1 / delta * ((-self._d2 * cos(x[2])) * (self._d2 * sin(x[2]) * x[3]**2 + self.input(t))
                 + (self._d0) * (self._d2 * self._g * sin(x[2])))]

    def simulate(self, initial_state, final_time, input=lambda t: 0, method="RK45", rtol=1e-9, atol=1e-9):
        """
        Solves the ode numerically starting from time = 0 
        and returns a dictionary with the results of the states
        
        Inputs::

        initial_state --
            Defines the initial state vector of type list [x (m), v (m/s), theta (rad), omega (rad/s)]

        final_time --
            Defines the end time of the simulation 
            units: (s)

        input --
            Passes the input function that will be called at each time step
            Default: 0

        method --
            Defines the numerical method to solve the ode
            Default: "RK45"

        rtol --
            Defines the relative tolerance that will be provided to the ode solver
            Default: 1e-9

        atol -- 
            Defines the absolute tolerance that will be provided to the ode solver
            Default: 1e-9

        Returns::

        results --
            Dictionary with keys of the type of output and values the output of the integration.
            e.x. : {"x" : list(), "theta" : list(), "time" : list()}
        """
        self.input = input
        solution = solve_ivp(self.ode, [self.time, final_time], initial_state, method, rtol=rtol, atol=atol)
        results = {"x" : solution.y[0], "theta" : solution.y[2], "time" : solution.t}
        self.states = [results["x"][-1], results["theta"][-1]]
        self.time = results["time"][-1]
        return results