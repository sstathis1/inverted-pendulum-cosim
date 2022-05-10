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

    def __init__(self, models, **kw):
        super().__init__()
        self.options = kw
        self.models = models
        self.check_names()

    def check_names(self):
        if self.models[0].name == self.models[1].name:
            self.models[0].name = self.models[0].name + "_0"
            self.models[1].name = self.models[1].name + "_1"

    def initialize(self, states):
        self.set_states(states)

    def get_outputs(self):
        output = []
        for model in self.models:
            output += list(model.output.values())
        return output

    def get_states(self):
        states = []
        for model in self.models:
            states += list(model.states.values())
        return states

    def set_states(self, states):
        i = 0
        for model in self.models:
            model.states = states[i:len(model.states)]
            i += len(model.states)

    def set_inputs(self, last_inputs):
        for model in self.models:
            model.input = self.extrapolate_input(last_inputs)

    def extrapolate_input(self, last_inputs):
        self._inputs += last_inputs

    def jacobi_algorithm(self, start_time, final_time):
        step_size = self.options["step_size"]
        if self.options["error_controlled"]:
            current_time = start_time
            while current_time < final_time:
                # Set states
                states = self.get_states()
                
                self.set_inputs()

                # Take a full step
                self.perform_step()
                y_full = self.get_outputs()

                # Restore states
                self.set_states(states)

                # Take a half step
                step_size = step_size / 2
                self.perform_step()

                # Take another half step
                self.perform_step()
                y_half = self.get_outputs()

                # Estimate error
                error = self.estimate_error(y_full, y_half)

                # Calculate next step size
                self.adapt_stepsize(error)
        else:
            pass

    def gauss_algorithm(self, start_time, final_time):
        pass

    def simulate(self, initial_states, start_time, final_time, **kw):
        self.options = kw
        self.initialize(initial_states)
        if self.options["communication_method"] == "Jacobi":
            self.jacobi_algorithm(start_time, final_time)
        elif self.options["communication_method"] == "Gauss":
            self.gauss_algorithm(start_time, final_time)
        else:
            raise Exception("A non valid communication method was passed")