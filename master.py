import numpy as np
from scipy.interpolate import interp1d
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
            "error_controlled" : False,
            "order" : 0,
            "is_parallel" : False,
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
        self._check_names()
        self.results = {}    

    def simulate(self, initial_states, start_time, final_time, **kw):
        self.options = kw
        self.results["time"] = start_time
        self._initialize(initial_states)
        if self.options["communication_method"] == "Jacobi":
            self._jacobi(start_time, final_time)
        elif self.options["communication_method"] == "Gauss":
            self._gauss(start_time, final_time)
        else:
            raise Exception("A non valid communication method was passed")
        return self.results

    def _check_names(self):
        if self.models[0].name == self.models[1].name:
            self.models[0].name = self.models[0].name + "_0"
            self.models[1].name = self.models[1].name + "_1"

    def _initialize(self, states):
        self._set_states(states)
        self._inputs = {}
        temp = {"values" : None, "time" : None}
        for model in self.models:
            self._inputs[model.name] = temp
            model.setup_experiment(self.options["step_size"])
            self._inputs[model.name]["values"] = list(model.input.values())
            self._inputs[model.name]["time"] = model.time
            for key in model.output:
                self.results[key] = model.output[key]    

    def _get_outputs(self):
        output = []
        for model in self.models:
            output += list(model.output.values())
        return output

    def _get_states(self):
        states = []
        for model in self.models:
            states += list(model.states.values())
        return states

    def _set_states(self, states):
        i = 0
        for model in self.models:
            model.states = states[i:len(model.states) + i]
            i += len(model.states)

    def _set_inputs(self, last_inputs, last_time):
        inputs = [None, None]
        inputs[0] = last_inputs[len(self.models[0].output)::]
        inputs[1] = last_inputs[0:len(self.models[0].output)]
        for i, model in enumerate(self.models):
            model.input = self._extrapolate(model.name, inputs[i], last_time)

    def _extrapolate(self, name, new_inputs, new_time):
        if len(self._inputs[name]["values"]) <= self.options["order"]:
            self._inputs[name]["values"] += new_inputs
            self._inputs[name]["time"] += new_time
        else:
            self._inputs[name]["values"].pop(0)
            self._inputs[name]["time"].pop(0)
            self._inputs[name]["values"] += new_inputs
            self._inputs[name]["time"] += new_time
        return interp1d(self._inputs[name]["time"], self._inputs[name]["values"], kind=self.options["order"])

    def _perform_step(self, step_size):
        if self.options["is_parallel"]:
            self._perform_step_parallel(step_size)
        else:
            for model in self.models:
                model.do_step(step_size)
                model.time += step_size

    def _perform_step_parallel(self, step_size):
        # TODO : Write an algorithm to perform the step in parallel for all models using multithreading
        pass

    def _jacobi(self, start_time, final_time):
        step_size = self.options["step_size"]
        if self.options["error_controlled"]:
            current_time = start_time
            while current_time < final_time:
                # Set states
                states = self._get_states()

                # Take a full step
                self._perform_step(step_size)
                y_full = self._get_outputs()
                current_time += step_size

                # Restore states
                self._set_states(states)

                # Take a half step
                step_size = step_size / 2
                self._perform_step(step_size)

                # Take another half step
                self._perform_step(step_size)
                y_half = self._get_outputs()

                # Estimate error
                error = self._estimate_error(y_full, y_half)
                
                # Calculate next step size
                step_size = self._adapt_stepsize(error)

                # Set inputs
                self._set_inputs(y_half, current_time)
        else:
            steps = int((final_time - start_time) / step_size) + 1
            time = np.linspace(start_time, final_time, steps)
            for t in time:
                # Take a full step
                self._perform_step(step_size)
                y = self._get_outputs()

                # Set inputs
                self._set_inputs(y, t)

                # Store the results
                self._set_results(t)

    def _gauss(self, start_time, final_time):
        # TODO : Write the algorithm for Gauss-Seidel co-simulation
        pass

    def _set_results(self, current_time):
        # Set the latest time in the results
        self.results["time"] += current_time

        # Set the latest outputs in the results
        for model in self.models:
            for key in model.output:
                self.results[key] += model.output[key]