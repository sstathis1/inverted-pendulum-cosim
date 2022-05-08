import time
import numpy as np
from timeit import default_timer as timer

class MasterOptions():
    """
    Master options for explicit co-simulation of slave models

    Options::

    local_rtol --
        Defines the relative tolerance that will be provided to the 
        connected models.
        Default: 1e-6

    rtol --
        Defines the relative tolerance used when an error controlled
        simulation is performed.
        Default: 1e-4

    atol --
        Defines the absolute tolerance used when an error controlled
        simulation is performed.
        Default: 1e-4

    step_size --
        Specfies the communication step-size to be used for simulating
        the coupled system.
        Default: 0.01

    step_max --
        Defines the maximum communication step-size that is allowed for
        the master.
        Default: 0.1

    step_min --
        Defines the minimum communication step-size that is allowed for
        the master.
        Default: 0.0001

    error_controlled --
        Specifies if richardson extrapolation will be used for controlling
        the error of the co-simulation.
        Default: True

    order --
        Defines the order for extrapolation / interpolation of inputs.
        Options: 1, 2, 3
        Default: 0

    is_parallel --
        Specifies if the models will be simulted on different threads.
        Default: True

    communication_method --
        Specifies the algorithm to use for the communication between the
        communication points.
        Options: "Jacobi", "Gauss"
        Default: "Jacobi"
    """
    def __init__(self):
        self._options = {
            "local_rtol" : 1e-6,
            "rtol" : 1e-4,
            "atol": 1e-4,
            "step_size" : 0.01,
            "step_max" : 0.1,
            "step_min" : 0.0001,
            "error_controlled" : True,
            "order" : 0,
            "is_parallel" : True,
            "communication_method" : "Jacobi"
        }

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, arguments):
        for key in arguments:
            if key in self._options:
                # Check if the step size given is within the max and min range
                if key == "step_size":
                    if arguments[key] > self._options["step_max"]:
                        self._options["step_max"] = arguments[key] * 10
                        print("Warning: maximum step is smaller than step size given."
                        " Setting step_max equal to step size * 10")
                    elif arguments[key] < self._options["step_min"]:
                        self._options["step_min"] = arguments[key] / 10
                        print("Warning: minimum step is bigger than step size given."
                        " Setting step_min equal to step size / 10")
                self._options[key] = arguments[key]


class Master(MasterOptions):
    """ 
    Master algorithm for explicit co-simulation of slave models.
    """

    def __init__(self, **kw):
        super().__init__()
        self.options = kw
        self.options = super().options

    def initialize(self, models):
        for model in models:
            self.outputs[model.name] = model.output
            self.states[model.name] = model.states