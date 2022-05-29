import numpy as np
from scipy.interpolate import interp1d
import threading

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
            "rtol" : 1e-6,
            "atol": 1e-6,
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
        """Simulates the master object using explicit co-simulation
        
        Parameters :
        ------------
        initial_states :
            A list containing the initial conditions for both models
        
        start_time :
            The starting time for the simulation (s)
        
        final_time :
            The final time for the simulation (s)

        **kw :
            keywords containing the master options to override

        Returns :
        ---------
            The results of the simulation in a dictionary with keys the name of the variables
            and values of lists containing the results of the models in [start_time, final_time] 
        """
        self.options = kw
        self.results["time"] = [start_time]
        self._initialize(start_time, initial_states)
        if self.options["communication_method"] == "Jacobi":
            self._jacobi(start_time, final_time)
        elif self.options["communication_method"] == "Gauss":
            self._gauss(start_time, final_time)
        else:
            raise Exception("A non valid communication method was passed")
        return self.results

    def _check_names(self):
        """Checks if two models share the same name and numbers them"""
        if self.models[0].name == self.models[1].name:
            self.models[0].name = self.models[0].name + "_0"
            self.models[1].name = self.models[1].name + "_1"

    def _initialize(self, start_time, states):
        """Initializes the co-simulation
        
        Parameters :
        ------------
        start_time :
            The starting time for the co-simulation (s)

        states :
            A list containing the initial states of all models at t = start_time
        """
        self._set_states(states)
        self._inputs = {}
        if self.options["error_controlled"]:
            self.results["error"] = {}
        for model in self.models:
            model.time = start_time
            for key in model.output:
                try:
                    self.results["error"][key] = [0]
                except KeyError:
                    pass
            self._inputs[model.name] = {"values" : None, "time" : None}
            model.setup_experiment(self.options["step_size"])
            self._inputs[model.name]["values"] = []
            self._inputs[model.name]["time"] = []
            for key in model.states:
                self.results[key] = np.array(model.states[key])
        y_initial = self._get_outputs()
        self._set_inputs(y_initial, start_time)

    def _get_outputs(self):
        """Returs a list containing the outputs of all model at a time instant"""
        output = []
        for model in self.models:
            for key, value in model.output.items():
                output.append(float(value))
        return output

    def _get_states(self):
        """Returns a list containing the states of all models at a time instant"""
        states = []
        for model in self.models:
            for key, value in model.states.items():
                states.append(float(value))
        return states

    def _set_states(self, states):
        """Set's the states for each model with the given new states
        
        Parameters :
        ------------
        states :
            A list containing the states of all models
        """
        i = 0
        for model in self.models:
            model.states = states[i:len(model.states) + i]
            i += len(model.states)

    def _set_inputs(self, last_inputs, last_time):
        """Set's the inputs of all models at a specific time instant"""
        inputs = [None, None]
        inputs[0] = last_inputs[len(self.models[0].output)::]
        inputs[1] = last_inputs[0:len(self.models[0].output)]
        for i, model in enumerate(self.models):
            model.input = self._extrapolate(model.name, inputs[i], last_time)

    def _extrapolate(self, name, new_inputs, new_time):
        """Extrapolates or Interpolates the inputs for each model
        
        Parameters :
        ------------
        name :
            The name of the model we want to extrapolate

        new_inputs :
            A list containing the values we want to add to the inputs

        new_time :
            The time at which the new_inputs belong (s)

        Returns :
        ---------
            A list containing the langrange polynomials of order equal to the order specified in settings
        """
        out = []
        if len(self._inputs[name]["values"]) < len(new_inputs) * (self.options["order"] + 1):
            self._inputs[name]["values"] += new_inputs
            self._inputs[name]["time"].append(new_time)
        else:
            for i in range(len(new_inputs)):
                self._inputs[name]["values"].pop(0)
            self._inputs[name]["time"].pop(0)
            self._inputs[name]["values"] += new_inputs
            self._inputs[name]["time"].append(new_time)
        for i in range(len(new_inputs)):
            tmp = self._inputs[name]["values"][i::len(new_inputs)]
            out.append(interp1d(self._inputs[name]["time"], tmp, kind=len(self._inputs[name]["time"]) - 1, 
                                fill_value="extrapolate"))
        return out

    def _perform_step(self, step_size, **kwargs):
        """
        Performs a Jacobi step either in parallel using multithreading 
        if is_parallel option is True or in serial.
        
        Parameters :
        ------------
        step_size :
            The current step_size to perform the step for (s)

        **kwargs :
            Aditional kwargs that may be required by a model in the model.do_step method

        """
        if self.options["is_parallel"]:
            self._perform_step_parallel(step_size)
        else:
            for model in self.models:
                model.do_step(step_size, **kwargs)

    def _perform_step_parallel(self, step_size):
        """Performs a Jacobi step in parallel using multithreading
        
        Parameters :
        ------------
        step_size :
            The size of the step that will be simulated (s)
        """
        threads = []
        for model in self.models:
            t = threading.Thread(target=model.do_step, args=(step_size, ))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()

    def _estimate_error(self, y_full, y_half):
        """Estimates the current local error using richardson extrapolation
        
        Parameters :
        ------------
        y_full :
            A list containing the outputs of all models performed with a full step

        y_half :
            A list containing the outputs of all models performed with two half steps

        Returns :
        ---------
            The current time local error due to co-simulation
        """
        order = self.options["order"]
        error = list(np.abs((np.array(y_half) - np.array(y_full))/(1-2**(order+1))))
        return error

    def _jacobi(self, start_time, final_time):
        """Performs the simulation from start_time to final_time using the Jacobi algorithm for data exchange
        
        Parameters :
        ------------
        start_time :
            The starting time of the simulation (s)

        final_time :
            The final time of the simulation (s)
        """
        step_size = self.options["step_size"]
        if self.options["error_controlled"]:
            current_time = start_time
            steps = int((final_time - start_time) / step_size) + 1
            time = np.linspace(start_time, final_time, steps)
            print_times = time[0::int(0.1 / step_size)]
            while current_time < final_time:
                if float(f"{current_time:.5f}") in print_times:
                    print(f"Solving at t = {current_time:.1f}...")
                # Get states
                states = self._get_states()

                # Take a full step
                self._perform_step(step_size, is_adapt=True)
                y_full = self._get_outputs()

                # Restore states
                self._set_states(states)

                # Take a half step
                step_size = step_size / 2
                self._perform_step(step_size, is_adapt=True)
                y_half = self._get_outputs()
                current_time += step_size

                # Update the time for each model
                for model in self.models:
                    model.time += step_size

                # Set inputs
                self._set_inputs(y_half, current_time)

                # Take another half step
                self._perform_step(step_size, is_adapt=True)
                y_half = self._get_outputs()
                current_time += step_size

                # Update the time for each model
                for model in self.models:
                    model.time += step_size
                    model.restore()

                # Estimate error
                error = self._estimate_error(y_full, y_half)

                # Calculate optimal step size
                step_size = step_size * 2

                # Set inputs
                self._set_inputs(y_half, current_time)

                # Store the results
                self._set_results(current_time, error=error)
        else:
            steps = int((final_time - start_time) / step_size) + 1
            time = np.linspace(start_time, final_time, steps)
            print_times = time[0::int(0.1 / step_size)]
            for t in time:
                if t in print_times:
                    print(f"Solving at t = {t:.1f}...")
                # Take a full step
                self._perform_step(step_size)
                y = self._get_outputs()
                current_time = t + step_size

                # Update the time for each model
                for model in self.models:
                    model.time += step_size

                # Set inputs
                self._set_inputs(y, current_time)

                # Store the results
                self._set_results(current_time)

    def _gauss(self, start_time, final_time):
        """Performs the simulation from start_time to final_time using the Gauss-Seidel algorithm for data exchange
        
        Parameters :
        ------------
        start_time :
            The starting time of the simulation (s)

        final_time :
            The final time of the simulation (s)
        """
        step_size = self.options["step_size"]
        if self.options["error_controlled"]:
            current_time = start_time
            steps = int((final_time - start_time) / step_size) + 1
            time = np.linspace(start_time, final_time, steps)
            print_times = time[0::int(0.1 / step_size)]
            while current_time < final_time:
                if float(f"{current_time:.5f}") in print_times:
                    print(f"Solving at t = {current_time:.1f}...")

                # Get states
                states = self._get_states()

                # Take a step for the first model
                self.models[0].do_step(step_size, is_adapt=True)

                # Store the output
                y1 = []
                for key, value in self.models[0].output.items():
                    y1.append(float(value))

                # Interpolate the input for the second model
                self.models[1].input = self._extrapolate(self.models[1].name, y1, current_time + step_size)

                # Take a step with the second model
                self.models[1].do_step(step_size, is_adapt=True)

                # Store the output
                y2 = []
                for key, value in self.models[1].output.items():
                    y2.append(float(value))

                # Extrapolate the input for the first model
                self.models[0].input = self._extrapolate(self.models[0].name, y2, current_time + step_size)

                y_full = y1 + y2

                # Restore states
                self._set_states(states)

                # Take a half step with the first model
                step_size = step_size / 2
                self.models[0].do_step(step_size, is_adapt=True)

                # Store the output
                y1 = []
                for key, value in self.models[0].output.items():
                    y1.append(float(value))

                # Interpolate the input for the second model
                self.models[1].input = self._extrapolate(self.models[1].name, y1, current_time + step_size)

                # Take a half step with the second model
                self.models[1].do_step(step_size, is_adapt=True)

                # Store the output
                y2 = []
                for key, value in self.models[1].output.items():
                    y2.append(float(value))

                # Extrapolate the input for the first model
                self.models[0].input = self._extrapolate(self.models[0].name, y2, current_time + step_size)

                # Update the time for each model
                for model in self.models:
                    model.time += step_size

                current_time += step_size

                # Take another half step with the first model
                self.models[0].do_step(step_size, is_adapt=True)

                # Store the output
                y1 = []
                for key, value in self.models[0].output.items():
                    y1.append(float(value))

                # Interpolate the input for the second model
                self.models[1].input = self._extrapolate(self.models[1].name, y1, current_time + step_size)

                # Take another half step with the second model
                self.models[1].do_step(step_size, is_adapt=True)

                # Store the output
                y2 = []
                for key, value in self.models[1].output.items():
                    y2.append(float(value))

                # Extrapolate the input for the first model
                self.models[0].input = self._extrapolate(self.models[0].name, y2, current_time + step_size)

                y_half = y1 + y2

                current_time += step_size
                # Update the time for each model
                for model in self.models:
                    model.time += step_size
                    model.restore()

                # Estimate error
                error = self._estimate_error(y_full, y_half)

                # Calculate optimal step size
                step_size = step_size * 2

                # Store the results
                self._set_results(current_time, error=error)
        else:
            steps = int((final_time - start_time) / step_size) + 1
            time = np.linspace(start_time, final_time, steps)
            print_times = time[0::int(0.1 / step_size)]
            for t in time:
                if t in print_times:
                    print(f"Solving at t = {t:.1f}...")

                # Take a step for the first model
                self.models[0].do_step(step_size)

                # Store the output
                y = []
                for key, value in self.models[0].output.items():
                    y.append(float(value))

                # Interpolate the input for the second model
                self.models[1].input = self._extrapolate(self.models[1].name, y, t + step_size)

                # Take a step with the second model
                self.models[1].do_step(step_size)

                # Store the output
                y = []
                for key, value in self.models[1].output.items():
                    y.append(float(value))

                # Update the time for each model
                for model in self.models:
                    model.time += step_size

                # Extrapolate the input for the first model
                self.models[0].input = self._extrapolate(self.models[0].name, y, t + step_size)

                # Store the results
                self._set_results(t + step_size)

    def _set_results(self, current_time, error=0):
        """
        Appends to the results the current_time and the current error if error_controlled option is set to True
        and also appends for each model the value of each state, output

        Parameters :
        ------------
        current_time :
            The time to append to the results (s)

        error :
            The current error computed using Richardson extrapolation if error_controlled option is set to True
            Default : 0 (error_controlled = False)
        """
        # Set the latest time in the results
        self.results["time"].append(current_time)

        # Set the latest outputs in the results
        i = 0
        for model in self.models:
            for key in model.output|model.states|model.input:
                if key not in self.results:
                    self.results[key] = np.array((model.output|model.states|model.input)[key])
                self.results[key] = np.append(self.results[key], (model.output|model.states|model.input)[key])
                # Set the latest error
                try:
                    self.results["error"][key].append(error[i])
                    i += 1
                except KeyError:
                    pass