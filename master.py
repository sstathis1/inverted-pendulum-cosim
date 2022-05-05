import time
import numpy as np
from timeit import default_timer as timer


class Model:
    """
    Creates a model for co-simulation.
    """

    def __init__(self, model):
        self.model = model
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        self.states = self.model.get_states()
        self.parameters = self.model.get_parameters()

    def get(self, string):
        for key in self.parameters:
            if key == string:
                return self.parameters[string]
        print("Warning: Could not find the specified parameter.")



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
    def __init__(self, **kw):
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
        for key in kw:
            if key in self._options:
                # Check if the step size given is within the max and min range
                if key == "step_size":
                    if kw[key] > self._options["step_max"]:
                        self._options["step_max"] = kw[key]
                        print("Warning: maximum step is smaller than step size given. Setting step_max equal to step size")
                    elif kw[key] < self._options["step_min"]:
                        self._options["step_min"] = kw[key]
                        print("Warning: minimum step is bigger than step size given. Setting step_min equal to step size")
                self._options[key] = kw[key]

    def get_options(self):
        return self._options


class Master(MasterOptions):
    """ 
    Master algorithm for explicit co-simulation of slave models.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.options = super().get_options()