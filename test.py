import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from master import Master
from double_pendulum import DoublePendulum as Pendulum
from double_pendulum_controller import DoublePendulumController as Controller

# Covariance Matrices
P = 10* np.eye(6)
Q = np.diag([0, 0, 0, 0, 0, 0])
R = np.diag([1e-4, 1e-4, 1e-4])

# Create the two model objects
model_1 = Pendulum(1.5, 0.5, 0.6, 0.4, 0.6, 0.05)
model_2 = Controller(1.5, 0.5, 0.6, 0.4, 0.6, 0.05, estimation_method="current", P=P, Q=Q, R=R)
models = [model_1, model_2]

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=0, communication_method="Gauss", 
                error_controlled=True, is_parallel=False)

# Simulate the models
start_time = 0
final_time = 5
initial_states = [0, 0, 20 * pi / 180, 0, 30 * pi / 180, 0, 0, 0, 20 * pi / 180, 0, 30 * pi / 180, 0]

# Start the timer
start_timer = time.perf_counter()

res = master.simulate(initial_states, start_time, final_time)

end_timer = time.perf_counter()
print(f"Co-Simulation finished correctly in : {end_timer-start_timer} second(s)")

model_1.animate(res, savefig=False)

states_nl = np.array([res["x"], res["v"], res["theta_1"], res["omega_1"], res["theta_2"], res["omega_2"]])
states_l = np.array([res["x_linear"], res["v_linear"], res["theta_1_linear"], res["omega_1_linear"], res["theta_2_linear"], res["omega_2_linear"]])

error_states = np.linalg.norm(states_nl - states_l, axis=0)

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["theta_1_linear"] * 180 / pi, label="linear")
plt.plot(res["time"], res["theta_1"] * 180 / pi, label="non-linear")
plt.plot(res["time"], res["theta_1_measured"] * 180 / pi, "*", label="measured", markersize=0.15)
plt.legend()
plt.ylabel("$θ_1$ (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("First Pendulum angle response")
plt.grid()
plt.show()

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["theta_2_linear"] * 180 / pi, label="linear")
plt.plot(res["time"], res["theta_2"] * 180 / pi, label="non-linear")
plt.plot(res["time"], res["theta_2_measured"] * 180 / pi, "*",label="measured", markersize=0.15)
plt.legend()
plt.ylabel("$θ_2$ (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Second Pendulum angle response")
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["x_linear"], label="linear")
plt.plot(res["time"], res["x"], label="non-linear")
plt.plot(res["time"], res["x_measured"], "*", label="measured", markersize=0.15)
plt.legend()
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Cart position response")
plt.grid()
plt.show()

# Plot the velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["v_linear"], label="linear")
plt.plot(res["time"], res["v"], label="non-linear")
plt.legend()
plt.ylabel("v (m/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Cart velocity response")
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["omega_1_linear"], label="linear")
plt.plot(res["time"], res["omega_1"], label="non-linear")
plt.legend()
plt.ylabel("$ω_1$ (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("First Pendulum rotational velocity response")
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["omega_2_linear"], label="linear")
plt.plot(res["time"], res["omega_2"], label="non-linear")
plt.legend()
plt.ylabel("$ω_2$ (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Second Pendulum rotational velocity response")
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["force"], label="force controller")
plt.step(res["time"], res["force_non_linear"], label="force non linear plant")
plt.legend()
plt.ylabel("force (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Double Pendulum on Cart required force (LQR)")
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], error_states)
plt.ylabel("state error (-)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("State error non-linear - linear")
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
plt.plot(res["time"], res["error"]["theta_1"])
plt.ylabel("$le^{θ_1}$ (rad)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of angle of first pendulum $(θ_1)$")
plt.grid()
plt.show()

# Plot the estimated error for angle of pendulum
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["theta_2"])
plt.ylabel("$le^{θ_2}$ (rad)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of angle of second pendulum $(θ_2)$")
plt.grid()
plt.show()

# Plot the estimated error for force by controller
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["force"])
plt.ylabel("$le^f$ (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Estimated local error of force from controller (f)")
plt.grid()
plt.show()