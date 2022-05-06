class Oscilator():
    """
    Initializes a one-degree of freedom linear oscilator object, 
    that can be actuated either by force or displacement input.

    Inputs::

    mass --
        Defines the mass of the oscilator (kg)

    stifness --
        Defines the stifness of the spring (N/m)

    damping --
        Defines the damping coefficient (Ns/m)

    output --
        Defines the type of output from the system
        Options: "force", "displacement"
        Default: "displacement"
    """
    def __init__(self, mass, stifness, damping, output="displacement"):
        self._name = "one-dof-oscilator"
        self._output = output
        self._m = mass
        self._k = stifness
        self._c = damping
        self._parameters = {"m" : self._m, "k" : self._k, "c" : self._c}

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def osc_method(self):
        return self._osc_method

    @osc_method.setter
    def osc_method(self, input):
        try:
            if len(input) == 2:
                self._osc_method = "displacement"
            elif len(input) == 1:
                self._osc_method = "force"
            else:
                raise Exception("Did not provide a valid input.")
        except TypeError:
            self._osc_method = "force"

    def get_inputs(self, time):
        """Returns the inputs that the model receives as strings"""
        return {self.input_name : self.input(time)}

    def set_input(self, input):
        """Sets the input for the system"""
        self.input_name, self.input = input

    def get_outputs(self):
        """Returns the outputs that the model produces as list of strings"""
        if self.__output == "displacement":
            return ["x", "v"]
        else:
            return "f"

    def get_states(self):
        """Returns the states of the model as a list of strings"""
        return ["x", "v"]

    def get(self, string):
        """Returns the value of the specified parameter via string if it exists else 0"""
        for key in self._parameters:
            if key == string:
                return self._parameters[string]
        print("Warning: Could not find the specified parameter.")
        return 0


    def setup_simulation(self, initial_state, input, start_time, final_time):
        pass
    
    def ode(self, t, x):
        """
        Contains the ordinary differential equation for the oscilator and
        returns the solution

        Inputs::
        t --
            time (s)
        x --
            state [x (m), v (m/s)]
        """
        return [x[1], -self._k/self._m*x[0] - self._c/self._m*x[1] + self.input(t)/self._m]