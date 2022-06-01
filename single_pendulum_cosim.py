import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from master import Master
from single_pendulum import SinglePendulum as Pendulum
from single_pendulum_controller import SinglePendulumController as Controller

# Covariance Matrices
P = 1e-2* np.eye(4)
Q = np.diag([0, 0, 0, 0, 0, 0])
R = np.diag([1e-4, 1e-4])

# Create the two model objects
model_1 = Pendulum(1.5, 0.5, 0.6, 0.05)
model_2 = Controller(1.5, 0.5, 0.6, 0.05, estimation_method="current", P=P, R=R)
models = [model_1, model_2]

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=0, communication_method="Jacobi", 
                error_controlled=True, is_parallel=False)

# Simulate the models
start_time = 0
final_time = 3
initial_states = [0, 0, 50 * pi / 180, 0, 0, 0, 50 * pi / 180, 0]

# Start the timer
start_timer = time.perf_counter()

res = master.simulate(initial_states, start_time, final_time)

end_timer = time.perf_counter()
print(f"Co-Simulation finished correctly in : {end_timer-start_timer} second(s)")

model_1.animate(res, savefig=False)

# Save the state error
states_nl = np.array([res["x"], res["v"], res["theta"], res["omega"]])
states_l = np.array([res["x_linear"], res["v_linear"], res["theta_linear"], res["omega_linear"]])

error_states = np.linalg.norm(states_nl - states_l, axis=0)

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["theta_linear"] * 180 / pi, label="γραμμική λύση")
plt.plot(res["time"], res["theta"] * 180 / pi, label="μη γραμμική λύση")
plt.plot(res["time"], res["theta_measured"] * 180 / pi, "*", label="μετρήσεις", markersize=0.15)
plt.legend()
plt.ylabel("$θ$ (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Απόκριση γωνίας θ εκκρεμούς (deg)")
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["x_linear"], label="γραμμική λύση")
plt.plot(res["time"], res["x"], label="μη γραμμική λύση")
plt.plot(res["time"], res["x_measured"], "*", label="μετρήσεις", markersize=0.15)
plt.legend()
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Απόκριση θέσης βαγονιού (m)")
plt.grid()
plt.show()

# Plot the velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["v_linear"], label="γραμμική λύση")
plt.plot(res["time"], res["v"], label="μη γραμμική λύση")
plt.legend()
plt.ylabel("v (m/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Απόκριση ταχύτητας βαγονιού (m/s)")
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["omega_linear"], label="γραμμική λύση")
plt.plot(res["time"], res["omega"], label="μη γραμμική λύση")
plt.legend()
plt.ylabel("$ω$ (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Απόκριση γωνιακής ταχύτητας εκκρεμούς (rad/s)")
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], res["force"], label="force controller", where="post")
plt.ylabel("u(t) (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Απαιτούμενη δύναμη για τον έλεγχο (Ν)")
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=200)
plt.step(res["time"], error_states, where="post")
plt.ylabel("$e^x$ (-)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Σφάλμα καταστάσεων μοντέλου - μη γραμμικού συστήματος")
plt.grid()
plt.show()

# Plot the estimated error for position of cart
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["x"])
plt.ylabel("$le^x$ (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Εκτιμώμενο τοπικό σφάλμα θέσης βαγονιού (m)")
plt.grid()
plt.show()

# Plot the estimated error for angle of pendulum
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["theta"])
plt.ylabel("$le^{θ}$ (rad)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Εκτιμώμενο τοπικό σφάλμα γωνίας εκκρεμούς (rad)")
plt.grid()
plt.show()

# Plot the estimated error for force by controller
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["error"]["force"])
plt.ylabel("$le^u$ (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, res["time"][-1])
plt.title("Εκτιμώμενο τοπικό σφάλμα απαιτούμενης δύναμης ελεγκτή (N)")
plt.grid()
plt.show()