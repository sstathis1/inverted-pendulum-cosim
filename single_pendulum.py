from numpy import cos, sin
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

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
    def __init__(self, mass_cart, mass_pendulum, length_pendulum):
        self._name = "single-inverted-pendulum"
        self._states = [None, None]
        self._output = [None, None]
        self._time = 0
        self._mc = mass_cart
        self._mp = mass_pendulum
        self._l = length_pendulum / 2
        self._I = (self._mp * (2 * self._l)**2) / 12
        self._g = 9.81
        self._d0 = self._mc + self._mp
        self._d1 = self._mp * self._l ** 2 + self._I
        self._d2 = self._mp * self._l
        self._parameters = {"mass_cart" : self._mc, "mass_pendulum" : self._mp, 
                            "length_pendulum" : 2 * self._l, "inertia_pendulum" : self._I}

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
        delta = self._d0 * self._d1 - (self._d2 * cos(x[2]))**2
        return  [x[1], 1 / delta * (self._d1 * (self.input(x) + self._d2 * x[3]**2 * sin(x[2]))
                 - self._d2**2 * cos(x[2]) * sin(x[2]) * self._g), x[3], 
                 1 / delta * (- self._d2 * cos(x[2]) * (self.input(x) + self._d2 * x[3]**2 * sin(x[2]))
                 + self._d0 * self._d2 * sin(x[2]) * self._g)]

    def simulate(self, initial_state, final_time, input=lambda t: 0, method="BDF", rtol=1e-12, atol=1e-9):
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
        solution = solve_ivp(self.ode, [self.time, final_time], initial_state, method, 
                             t_eval=np.linspace(0, final_time, 50*final_time), rtol=rtol, atol=atol)
        results = {"x" : solution.y[0], "theta" : solution.y[2], "v" : solution.y[1], 
                   "omega" : solution.y[3], "time" : solution.t}
        self.states = [results["x"][-1], results["theta"][-1]]
        self.time = results["time"][-1]
        return results

    def animate(self, results, savefig=True):
        """
        Animates the solution of the pendulum on the cart using matplotlib
        
        Input::

        results --
            The results of the simulation as a dictionary {"variable" : value}
        """
        def animations(i):
            x = [x_cart[i], x_pendulum[i]]
            y = [0, y_pendulum[i]]
            if i == 0:
                history_x.clear()
                history_y.clear()
                history_cart.clear()

            history_x.appendleft(x[1])
            history_y.appendleft(y[1])
            history_cart.appendleft(x[0])
            line.set_data(x, y)
            trace.set_data(history_x, history_y)
            trace_cart.set_data(history_cart, 0)
            time_text.set_text(time_template % (time[i]))
            return line

        history_len = 150  # how many trajectory points to display
        dt = results["time"][-1] / (50 * results["time"][-1])
        x_cart = results["x"]
        x_pendulum = x_cart + self.get("length_pendulum") * sin(results["theta"])
        y_pendulum = self.get("length_pendulum") * cos(results["theta"])
        time = results["time"]

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(xlim=(- 0.4 + np.min(x_pendulum), 0.4 + np.max(x_pendulum)), 
                             ylim=(- 0.1 + np.min(y_pendulum), 0.1 + np.max(y_pendulum)))
        ax.grid()
        line, = ax.plot([], [], "o-", lw=2)
        trace, = ax.plot([], [], '.-', lw=1, ms=2)
        trace_cart, = ax.plot([], [], '.-', lw=1, ms=2)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)
        history_cart = deque(maxlen=history_len)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ani = FuncAnimation(fig, animations, len(time), interval=dt*1000)
        plt.show()

        # Save animation in gif format
        if savefig:
            ani.save("single_pendulum.gif", writer='pillow')