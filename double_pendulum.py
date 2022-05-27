from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_ivp

class DoublePendulum():
    """Initializes a single inverted pendulum object

    Parameters
    ----------

    mass_cart :
        Defines the mass of the cart (kg)

    mass_pendulum_1 :
        Defines the mass of the first pendulum (kg)

    mass_pendulum_2 :
        Defines the mass of the second pendulum (kg)

    length_pendulum_1 :
        Defines the length of the first pendulum (m)

    length_pendulum_2 :
        Defines the length of the second pendulum (m)

    friction_coefficient :
        Defines the friction coefficient between the cart and the ground
    """
    def __init__(self, mass_cart, mass_pendulum_1, mass_pendulum_2, 
                 length_pendulum_1, length_pendulum_2, friction_coefficient):
        self._name = "double-inverted-pendulum"
        self._states = [None, None, None, None, None, None]
        self._output = [None, None, None]
        self._input = lambda t : 0
        self._time = 0
        self._m0 = mass_cart
        self._m1 = mass_pendulum_1
        self._m2 = mass_pendulum_2
        self._L1 = length_pendulum_1
        self._l1 = length_pendulum_1 / 2
        self._L2 = length_pendulum_2
        self._l2 = length_pendulum_2 / 2
        self._b = friction_coefficient
        self._I1 = (self._m1 * self._L1**2) / 12
        self._I2 = (self._m2 * self._L2**2) / 12
        self._g = 9.814
        self._d1 = self._m0 + self._m1 + self._m2
        self._d2 = self._m1 * self._l1 + self._m2 * self._L1
        self._d3 = self._m2 * self._l2
        self._d4 = self._m1 * self._l1**2 + self._m2 * self._L1**2 + self._I1
        self._d5 = self._m2 * self._L1 * self._l2
        self._d6 = self._m2 * self._l2**2 + self._I2
        self._f1 = (self._m1 * self._l1 + self._m2 * self._L1) * self._g
        self._f2 = self._m2 * self._l2 * self._g
        self._C = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
        self._parameters = {"mass_cart" : self._m0, "mass_pendulum_1" : self._m1, "mass_pendulum_2" : self._m2,
                            "length_pendulum_1" : self._L1, "length_pendulum_2" : self._L2, 
                            "inertia_pendulum_1" : self._I1, "inertia_pendulum_2" : self._I2,
                            "friction_coefficient" : self._b}

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

    @property
    def states(self):
        return {"x" : self._states[0], "v" : self._states[1], 
                "theta_1" : self._states[2], "omega_1" : self._states[3],
                "theta_2" : self._states[4], "omega_2" : self._states[5]}

    @states.setter
    def states(self, new):
        self._states = new

    @property
    def output(self):
        self._output = self._C.dot(self._states)
        return {"x" : self._output[0], "theta_1" : self._output[1], "theta_2" : self._output[2]}

    @output.setter
    def output(self, new):
        self._output = new

    @property
    def input(self):
        return {"force" : self._input(self.time)}

    @input.setter
    def input(self, new):
        self._input = new[0]

    def setup_experiment(self, *args):
        self.output = self._C.dot(self._states)
    
    def restore(self):
        pass

    def get(self, string):
        """Returns the value of the specified parameter via string if it exists else 0"""
        for key in self._parameters:
            if key == string:
                return self._parameters[string]
        print("Warning: Could not find the specified parameter.")
        return 0

    def do_step(self, macro_step, **kwargs):
        """Does one step when called from the master object and returns True if it succeeded"""
        final_time = self.time + macro_step
        solution = solve_ivp(self._ode, [self.time, final_time], self._states, "BDF", rtol=1e-9, atol=1e-9)
        self.states = [solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], 
                       solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]]
        return True

    def simulate(self, initial_state, final_time, input=lambda t: 0, method="BDF", rtol=1e-12, atol=1e-9):
        """Solves the ode numerically starting from time = 0 
        and returns a dictionary with the results of the states
        
        Parameters
        ----------

        initial_state :
            Defines the initial state vector of type list [x (m), v (m/s), theta (rad), omega (rad/s)]

        final_time :
            Defines the end time of the simulation 
            units: (s)

        input --
            Passes the input function that will be called at each time step
            Default: 0

        method :
            Defines the numerical method to solve the ode
            Default: "RK45"

        rtol :
            Defines the relative tolerance that will be provided to the ode solver
            Default: 1e-9

        atol :
            Defines the absolute tolerance that will be provided to the ode solver
            Default: 1e-9

        Returns
        -------

        results :
            Dictionary with keys of the type of output and values the output of the integration.
            e.x. : {"x" : list(), "theta_1" : list(), "theta_2" : list(), "v" : list(), 
                    "omega_1" : list(), "omega_2" : list(), time" : list()}
        """
        self.input = [input]
        solution = solve_ivp(self._ode, [self.time, final_time], initial_state, method, 
                             t_eval=np.linspace(0, final_time, 50*final_time), rtol=rtol, atol=atol)
        results = {"x" : solution.y[0], "theta_1" : solution.y[2], "theta_2" : solution.y[4], 
                   "v" : solution.y[1], "omega_1" : solution.y[3], "omega_2" : solution.y[5], "time" : solution.t}
        # results["force"] = list(self._input([results["x"], results["v"], results["theta_1"], 
        #                                      results["omega_1"], results["theta_2"], results["omega_2"]])[0])
        self.states = [results["x"][-1], results["v"][-1], results["theta_1"][-1], results["omega_1"][-1],
                       results["theta_2"][-1], results["omega_2"][-1]]
        self.time = results["time"][-1]
        return results

    def animate(self, results, savefig=True):
        """Animates the solution of the pendulum on the cart using matplotlib
        
        Input
        -----

        results :
            The results of the simulation as a dictionary {"variable" : value}

        savefig :
            Boolean if True it saves a gif of the animation produced
            Default: True
        """
        def _animations(i):
            x = [x_cart[i], x_pendulum_1[i], x_pendulum_2[i]]
            y = [0, y_pendulum_1[i], y_pendulum_2[i]]
            if i == 0:
                history_px1.clear()
                history_py1.clear()
                history_px2.clear()
                history_py2.clear()
                history_cart.clear()

            history_px1.appendleft(x[1])
            history_py1.appendleft(y[1])
            history_px2.appendleft(x[2])
            history_py2.appendleft(y[2])
            history_cart.appendleft(x[0])
            line.set_data(x, y)
            trace_pendulum_1.set_data(history_px1, history_py1)
            trace_pendulum_2.set_data(history_px2, history_py2)
            trace_cart.set_data(history_cart, 0)
            time_text.set_text(time_template % (time[i]))
            return line, trace_pendulum_1, trace_pendulum_2, trace_cart
        
        # Defines how many data points to show at a time
        history_len = 1500
        
        # Time between two points in (s)
        dt = 0.001

        # x, y, time data from results for pendulum and cart
        x_cart = results["x"]
        x_pendulum_1 = x_cart + self.get("length_pendulum_1") * sin(results["theta_1"])
        y_pendulum_1 = self.get("length_pendulum_1") * cos(results["theta_1"])
        x_pendulum_2 = x_pendulum_1 + self.get("length_pendulum_2") * sin(results["theta_2"])
        y_pendulum_2 = y_pendulum_1 + self.get("length_pendulum_2") * cos(results["theta_2"])
        time = results["time"]

        # Create the figure
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(xlim=(- 0.2 + np.min(x_pendulum_2), 0.2 + np.max(x_pendulum_2)), 
                             ylim=(- 0.5 + np.min(x_pendulum_2), 0.2 + np.max(x_pendulum_2)))
        ax.grid()
        line, = ax.plot([], [], "o-", lw=4)
        trace_pendulum_1, = ax.plot([], [], '.-', lw=1, ms=2)
        trace_pendulum_2, = ax.plot([], [], '.-', lw=1, ms=2)
        trace_cart, = ax.plot([], [], '.-', lw=1, ms=2)
        history_px1, history_py1= deque(maxlen=history_len), deque(maxlen=history_len)
        history_px2, history_py2= deque(maxlen=history_len), deque(maxlen=history_len)
        history_cart = deque(maxlen=history_len)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, weight="bold")
        ani = FuncAnimation(fig, _animations, len(time), interval=dt*1000)
        plt.show()

        # Save animation in gif format
        if savefig:
            ani.save("double_pendulum.gif", writer='pillow')

    def _ode(self, t, x):
        """Contains the system of ordinary differential equations for the double-pendulum on a cart.

            Parameters
            ----------

            t :
                time (s)
            x :
                state [x (m), v (m/s), theta_1 (rad), omega_1 (rad/s), theta_2 (rad), omega_2 (rad/s)]
        """
        M = np.array([[self._d1, self._d2 * cos(x[2]), self._d3 * cos(x[4])],
                      [self._d2 * cos(x[2]), self._d4, self._d5 * cos(x[2] - x[4])],
                      [self._d3 * cos(x[4]), self._d5 * cos(x[2] - x[4]), self._d6]])
        
        C = np.array([[- self._d2 * sin(x[2]) * x[3]**2 - self._d3 * sin(x[4]) * x[5]**2 - self._input(t) + self._b * x[1]],
                      [self._d5 * sin(x[2] - x[4]) * x[5]**2 - self._f1 * sin(x[2])],
                      [- self._d5 * sin(x[2] - x[4]) * x[3]**2 - self._f2 * sin(x[4])]])

        tmp = - np.linalg.inv(M).dot(C)

        return [x[1], tmp[0, 0], x[3], tmp[1, 0], x[5], tmp[2, 0]]
