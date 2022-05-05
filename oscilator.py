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

    osc_method --
        Defines how the oscilation is actuated 
        Options: "force", "displacement"
        Default: "force"
        
    output --
        Defines the type of output from the system
        Options: "force", "displacement"
        Default: "displacement"
    """
    def __init__(self, mass, stifness, damping, osc_method="force", output="displacement"):
        self.__name = "one-dof-oscilator"
        self.__osc_method = osc_method
        self.__output = output
        self.m = mass
        self.k = stifness
        self.c = damping
        self.parameters = {"m" : self.m, "k" : self.k, "c" : self.c}

    def get_name(self):
        """Returns the name of the object"""
        return self.__name

    def set_name(self, name):
        """Set's a new name for the object"""
        self.__name = name

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
        for key in self.parameters:
            if key == string:
                return self.parameters[string]
        print("Warning: Could not find the specified parameter.")
        return 0

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
        return [x[1], -self.k/self.m*x[0] - self.c/self.m*x[1] + self.input(t)/self.m]