import control
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_ivp
from scipy import signal, linalg

class DoublePendulumController():
    """
    Initializes a double inverted pendulum controller object

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

    controller :
        Defines the controller used

        Options :
        "lqr" : solves the discrete algebraic ricatti equation
        "sdre" : solves the state dependend ricatti equation

    estimation_method :
        Defines how the controller will estimate the unkown states
        
        Options :
        "current" : uses a current estimator
        "predictive" : uses a predictive estimator
        "kalman" : uses kalman filter. Must also provide P, Q, R covariance matrices

    P :
        Covariance matrix of states at time 0.
        Use Gaussian white noise for optimal estimator
        Default : zero covariances, deterministic states

    Q :
        Covariance matrix of plant disturbances.
        Constant white noise.
        Default : zero covariances, deterministic plant

    R :
        Covariance matrix of measurment error.
        Constant white noise.
        Default : zero covariances, deterministic measurments
    """
    def __init__(self, mass_cart, mass_pendulum_1, mass_pendulum_2, length_pendulum_1, 
                 length_pendulum_2, friction_coefficient, controller="lqr", estimation_method="current", 
                 P=np.ones([6, 6]), Q=np.zeros([6, 6]), R=np.zeros([3, 3])):
        self._name = "double-inverted-pendulum-controller"
        self._states = [None, None, None, None, None, None]
        self._measurments = [None, None, None]
        self._estimation_method = estimation_method
        self._output = 0
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
        self._P = P
        self._P_prev = P
        self._Q = Q
        self._R = R
        self._parameters = {"mass_cart" : self._m0, "mass_pendulum_1" : self._m1, "mass_pendulum_2" : self._m2,
                            "length_pendulum_1" : self._L1, "length_pendulum_2" : self._L2, 
                            "inertia_pendulum_1" : self._I1, "inertia_pendulum_2" : self._I2,
                            "friction_coefficient" : self._b}
        self._create_ss_matrices()
        if not self._is_observable():
            raise Exception("The system is not observable with the given measurments and canot be estimated.")

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
    def input(self):
        return {"x_measured" : self._measurments[0], "theta_1_measured" : self._measurments[1], "theta_2_measured" : self._measurments[2]}

    @input.setter
    def input(self, new):
        for i in range(len(new)):
            self._measurments[i] = float(new[i](self.time))

    @property
    def output(self):
        return {"force" : self._output}

    @output.setter
    def output(self, new):
        self._output = new

    @property
    def states(self):
        return {"x_linear" : self._states[0], "v_linear" : self._states[1], 
                "theta_1_linear" : self._states[2], "omega_1_linear" : self._states[3],
                "theta_2_linear" : self._states[4], "omega_2_linear" : self._states[5]}

    @states.setter
    def states(self, new):
        self._P = self._P_prev
        self._states = new

    @property
    def kalman_gain(self):
        return self._P.dot(self._Cd.T).dot(linalg.inv(self._Cd.dot(self._P).dot(self._Cd.T) + self._R))

    def setup_experiment(self, step_size):
        """ Initializes the experiment for co-simulation
        Will be called from the Master object.

        Parameters :
        ------------
        step_size : 
            The step used for co-simulation. Is also equal to the sampling time
        """
        self.sampling_time = step_size
        self._discretize_ss(step_size)

        # Compute the discretized covariances
        if self._estimation_method == "kalman":
            self._Q = self._Q * self.sampling_time
            self._R = self._R / self.sampling_time

        # Compute the optimal gain from discrete LQR
        self.gain = self._lqr()
        self.feedback = - self.gain.dot(self._states)
        self.output = self.feedback

        # Compute the predictive estimator gain using pole placement
        poles = [0.2, 0.1, 0.1 + 0.1j, 0.1 - 0.1j, 0.05 +0.05j, 0.05 - 0.05j]
        self._L = control.place(self._Ad.T, self._Cd.T, poles).T

    def restore(self):
        """Restores the previous state covariance matrix to the latest"""
        if self._estimation_method == "kalman":
            self._P_prev = self._P

    def get(self, string):
        """Returns the value of the specified parameter via string if it exists else 0"""
        for key in self._parameters:
            if key == string:
                return self._parameters[string]
        print("Warning: Could not find the specified parameter.")
        return 0

    def do_step(self, step_size, is_adapt=False):
        """Does one step when called from the master object and returns True if it succeeded"""
        if is_adapt:
            self._discretize_ss(step_size)
        self._predict()
        self._correct()
        self.feedback = - self.gain.dot(self._states)
        self.output = - self.gain.dot(self._states)
        return True

    def simulate(self, initial_state, final_time, input=lambda t: 0, method="RK45", rtol=1e-12, atol=1e-9):
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
            e.x. : {"x" : list(), "theta" : list(), "time" : list()}
        """
        K = self._lqr()
        self.u = lambda x: -K.dot(x)
        solution = solve_ivp(self._ode, [self.time, final_time], initial_state, method, 
                             t_eval=np.linspace(0, final_time, int(50*final_time)), rtol=rtol, atol=atol)
        results = {"x" : solution.y[0], "theta_1" : solution.y[2], "theta_2" : solution.y[4], 
                   "v" : solution.y[1], "omega_1" : solution.y[3], "omega_2" : solution.y[5], "time" : solution.t}
        # results["force"] = list(self._input([results["x"], results["v"], results["theta_1"], 
        #                                      results["omega_1"], results["theta_2"], results["omega_2"]])[0])
        self.states = [results["x"][-1], results["v"][-1], results["theta_1"][-1], results["omega_1"][-1],
                       results["theta_2"][-1], results["omega_2"][-1]]
        self.time = results["time"][-1]
        return results

    def animate(self, results, savefig=True):
        """Animates the solution of the double pendulum on the cart using matplotlib
        
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
        x_cart = results["x"][0::10]
        x_pendulum_1 = x_cart + self.get("length_pendulum_1") * sin(results["theta_1"][0::10])
        y_pendulum_1 = self.get("length_pendulum_1") * cos(results["theta_1"][0::10])
        x_pendulum_2 = x_pendulum_1 + self.get("length_pendulum_2") * sin(results["theta_2"][0::10])
        y_pendulum_2 = y_pendulum_1 + self.get("length_pendulum_2") * cos(results["theta_2"][0::10])
        time = results["time"][0::10]

        # Create the figure
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(xlim=(- 0.2 + np.min(x_pendulum_2), 0.2 + np.max(x_pendulum_2)), 
                             ylim=(- 0.75 + np.min(y_pendulum_1), 0.75 + np.max(y_pendulum_2)))
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
        """Contains the linearized system of ordinary differential equations for the single-pendulum on a cart.

        Parameters
        ----------

        t :
            time (s)
        x :
            state [x (m), v (m/s), theta_1 (rad), omega_1 (rad/s), theta_2 (rad), omega_2 (rad/s)]
        """
        return list(self._A.dot(x) + self._B.dot(self.u(x)))

    def _lqr(self):
        """Solves the discrete algebraic ricatti equation for optimal LQR control
        
        Returns :
        ---------
        K : Gain Matrix (u = -K * x)
        """
        Q = np.diag([200, 50, 200, 700, 200, 700])
        R = 1
        P = linalg.solve_discrete_are(self._Ad, self._Bd, Q, R)
        return 1 / R * self._Bd.T.dot(P)

    def _is_observable(self):
        """Checks whether the system is observable given matrices A, C
        
        Returns
        -------
            True: if the system is observable, else False
        """
        if np.linalg.matrix_rank(control.obsv(self._A, self._C)) == len(self._A):
            return True
        else:
            return False

    def _create_ss_matrices(self):
        """Creates the continuous-time state-space matrices of the linear model"""
        C = np.array([[self._b, 0, 0], [0, 0, 0], [0, 0, 0]])
        K = np.array([[0, 0, 0], [0, -self._f1, 0], [0, 0, -self._f2]])
        M = np.array([[self._d1, self._d2, self._d3],
                      [self._d2, self._d4, self._d5],
                      [self._d3, self._d5, self._d6]])
        H = np.array([[1], [0], [0]])
        M_inverse = np.linalg.inv(M)
        tmp1 = - M_inverse.dot(C)
        tmp2 = - M_inverse.dot(K)
        tmp3 = M_inverse.dot(H)
        self._A = np.array([[0, 1, 0, 0, 0, 0],
                            [tmp2[0, 0], tmp1[0, 0], tmp2[0, 1], tmp1[0, 1], tmp2[0, 2], tmp1[0, 2]],
                            [0, 0, 0, 1, 0, 0],
                            [tmp2[1, 0], tmp1[1, 0], tmp2[1, 1], tmp1[1, 1], tmp2[1, 2], tmp1[1, 2]],
                            [0, 0, 0, 0, 0, 1],
                            [tmp2[2, 0], tmp1[2, 0], tmp2[2, 1], tmp1[2, 1], tmp2[2, 2], tmp1[2, 2]]])
        self._B = np.array([0, tmp3[0, 0], 0, tmp3[1, 0], 0, tmp3[2, 0]]).reshape(-1, 1)
        self._C = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
        self._D = np.zeros([3, 1])

    def _discretize_ss(self, step_size):
        """Creates the discrete-time state-space matrices with the given sampling time"""
        self._Ad, self._Bd, self._Cd, self._Dd, _ = signal.cont2discrete((self._A, self._B, self._C, self._D), 
                                                                          step_size)

    def _predict(self):
        """Makes a prediction for the states and covariances if kalman filter is used for estimation"""
        self._predict_states()
        if self._estimation_method == "kalman":
            self._predict_covariance()

    def _predict_states(self):
        """Predicts the states using the linear model"""
        self.feedback = - self.gain.dot(self._states)
        if self._estimation_method == "predictive":
            self._states = (self._Ad.dot(self._states) + self._Bd.dot(self.feedback) 
                           + self._L.dot(self._measurments - self._Cd.dot(self._states)))
        else:
            self._states = self._Ad.dot(self._states) + self._Bd.dot(self.feedback)

    def _predict_covariance(self):
        """Predicts the state covariance matrix if kalman filter is used for estimation"""
        self._P = self._Ad.dot(self._P).dot(self._Ad.T) + self._Q

    def _correct(self):
        """Corrects the states and covariances if kalman filter is used for estimation"""
        if self._estimation_method == "kalman":
            gain = self.kalman_gain
            self._correct_covariance(gain)
            self._correct_states(gain)
        elif self._estimation_method == "predictive":
            pass
        elif self._estimation_method == "current":
            gain = np.linalg.inv(self._Ad).dot(self._L)
            self._correct_states(gain)
        else:
            raise Exception("Estimation method provided is not a valid one \n"
                            "Options: 1.'kalman', 2.'current', 3.'predictive'")

    def _correct_states(self, gain):
        """Corrects the states based on the measurments taken and the appropriate gain
        
        Parameters :
        ------------

        gain :
            The gain used for estimating. ->  L*(z - C*x)
        """
        self._states = self._states + gain.dot(self._measurments - self._Cd.dot(self._states))

    def _correct_covariance(self, gain):
        """Corrects the state covariance matrix if kalman filter is used for estimation"""
        self._P = ((np.eye(6) - gain.dot(self._Cd)).dot(self._P).dot((np.eye(6) - gain.dot(self._Cd)).T) 
                    + gain.dot(self._R).dot(gain.T))