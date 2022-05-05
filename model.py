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

    def get_name(self):
        """Returns the name of the object"""
        return self.__name

    def set_name(self, name):
        """Set's a new name for the object"""
        self.__name = name

    def get_inputs(self):
        return self.__osc_method

    def get_outputs(self):
        if self.__output == "displacement":
            return ["x", "v"]
        else:
            return "f"

    def get_states(self):
        return ["x", "v"]

    def get_parameters(self):
        return {"m" : self.m, "k": self.k, "c": self.c}