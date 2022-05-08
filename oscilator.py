from scipy.integrate import solve_ivp

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
    def __init__(self, mass, stifness, damping):
        self._name = "one-dof-oscilator"
        self._states = [None, None]
        self._output = [None, None]
        self._time = 0
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

    @property
    def input(self):
        return {self.osc_method : self._input(self.time)}

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def output(self):
        return self.states

    @output.setter
    def output(self, new):
        self._output = new

    @property
    def states(self):
        return {"x" : self._states[0], "v" : self._states[1]}

    @states.setter
    def states(self, new):
        self._states = new

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
        Contains the ordinary differential equations for the oscilator and
        returns the solution

        Inputs::
        t --
            time (s)
        x --
            state [x (m), v (m/s)]
        """
        if self.osc_method == "force":
            return [x[1], -self._k/self._m*x[0] - self._c/self._m*x[1] + self._input(t)/self._m]
        else:
            return ([x[1], -(2*self._k)/self._m*x[0] -(2*self._c)/self._m*x[1] 
                + self._k/self._m*self._input(t)[0] + self._c/self._m*self._input(t)[1]]) 

    def simulate(self, initial_state, input, final_time, method="RK45", rtol=1e-9, atol=1e-9):
        """
        Solves the ode numerically starting from time = 0 
        and returns a dictionary with the results of the states
        
        Inputs::

        initial_state --
            Defines the initial state vector of type list [x (m), v (m/s)]

        input --
            Passes the input function that will be called at each time step

        final_time --
            Defines the end time of the simulation 
            units: (s)

        Returns::

        results --
            Dictionary with keys of the type of output and values the output of the integration.
            e.x. : {"x" : list(), "v" : list(), "time" : list()}
        """
        self.input = input
        self.osc_method = input(0)
        solution = solve_ivp(self.ode, [self.time, final_time], initial_state, method, rtol=rtol, atol=atol)
        results = {"x" : solution.y[0], "v" : solution.y[1], "time" : solution.t}
        self.states = [results["x"][-1], results["v"][-1]]
        self.time = results["time"][-1]
        return results