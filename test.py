import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from master import Master
from single_pendulum import SinglePendulum as Pendulum
from single_pendulum_controller import SinglePendulumController as Controller

# Covariance Matrices
P = 1 * np.eye(4)
Q = np.diag([0, 0, 1e-5, 0])
R = np.array([[1e-3, 0], [0, 1e-3]])

# Create the two model objects
model_2 = Pendulum(1.5, 0.2, 0.4, 0.05)
model_1 = Controller(1.5, 0.2, 0.4, 0.05, P=P, Q=Q, R=R)
models = [model_1, model_2]

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=0, communication_method="Jacobi", 
                error_controlled=False, is_parallel=False)

# Simulate the models
start_time = 0
final_time = 7
initial_states = [0, 0, 45 * pi / 180, 0, 0, 0, 45 * pi / 180, 0]

# Start the timer
start_timer = time.perf_counter()

res = master.simulate(initial_states, start_time, final_time)

end_timer = time.perf_counter()
print(f"Co-Simulation finished correctly in : {end_timer-start_timer} second(s)")

model_1.animate(res, savefig=False)

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["theta_linear"] * 180 / pi, label="linear")
plt.plot(res["time"], res["theta"] * 180 / pi, label="non-linear")
plt.legend()
plt.ylabel("theta (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Single Pendulum on Cart angle response")
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["x_linear"], label="linear")
plt.plot(res["time"], res["x"], label="non-linear")
plt.legend()
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Single Pendulum on Cart position response")
plt.grid()
plt.show()

# Plot the velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["v_linear"], label="linear")
plt.plot(res["time"], res["v"], label="non-linear")
plt.legend()
plt.ylabel("v (m/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Single Pendulum on Cart velocity response")
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["omega_linear"], label="linear")
plt.plot(res["time"], res["omega"], label="non-linear")
plt.legend()
plt.ylabel("omega (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Single Pendulum on Cart rotational velocity response")
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["force"])
plt.ylabel("force (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Single Pendulum on Cart required force (LQR)")
plt.grid()
plt.show()

# Plot the estimated error for position of cart
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["x"])
plt.ylabel("$le^x$ (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of position of cart (x)")
plt.grid()
plt.show()

# Plot the estimated error for angle of pendulum
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["theta"])
plt.ylabel("$le^{θ}$ (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of angle of pendulum (θ)")
plt.grid()
plt.show()

# Plot the estimated error for force by controller
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["force"])
plt.ylabel("$le^f$ (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of force from controller (f)")
plt.grid()
plt.show()