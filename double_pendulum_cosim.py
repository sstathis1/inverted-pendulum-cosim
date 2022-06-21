import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from master import Master
from double_pendulum import DoublePendulum as Pendulum
from double_pendulum_controller import DoublePendulumController as Controller

# Covariance Matrices
P = 1e-3* np.eye(6)
Q = np.diag([0, 0, 0, 0, 0, 0, 0, 0])
R = np.diag([1e-4, 1e-4, 1e-4])

# Create the two model objects
model_1 = Pendulum(1.5, 0.5, 0.6, 0.4, 0.6, 0.05)
model_2 = Controller(1.5, 0.5, 0.6, 0.4, 0.6, 0.05, estimation_method="current", P=P, R=R)
models = [model_1, model_2]

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=0, communication_method="Gauss", 
                error_controlled=False, is_parallel=False)

# Simulate the models
start_time = 0
final_time = 5
initial_states = [0, 0, 20 * pi / 180, 0, 30 * pi / 180, 0, 0, 0, 20 * pi / 180, 0, 30 * pi / 180, 0]

# Start the timer
start_timer = time.perf_counter()

res = master.simulate(initial_states, start_time, final_time)

end_timer = time.perf_counter()
print(f"Co-Simulation finished correctly in : {end_timer-start_timer} second(s)")

model_1.animate(res, savefig=True)

# Save the state error
states_nl = np.array([res["x"], res["v"], res["theta_1"], res["omega_1"], res["theta_2"], res["omega_2"]])
states_l = np.array([res["x_linear"], res["v_linear"], res["theta_1_linear"], res["omega_1_linear"], 
                     res["theta_2_linear"], res["omega_2_linear"]])

error_states = np.linalg.norm(states_nl - states_l, axis=0)

# Plot Options
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE, weight = 'bold')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, labelweight = 'bold')     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight = 'bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["theta_1_linear"] * 180 / pi, label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["theta_1"] * 180 / pi, label="μη γραμμική λύση", linewidth=2.5)
plt.plot(res["time"], res["theta_1_measured"] * 180 / pi, "*", label="μετρήσεις", markersize=0.5)
plt.legend()
plt.ylabel("$θ_1$ (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση γωνίας $θ_1$ εκκρεμούς (deg)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["theta_2_linear"] * 180 / pi, label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["theta_2"] * 180 / pi, label="μη γραμμική λύση", linewidth=2.5)
plt.plot(res["time"], res["theta_2_measured"] * 180 / pi, "*", label="μετρήσεις", markersize=0.5)
plt.legend()
plt.ylabel("$θ_2$ (deg)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση γωνίας $θ_2$ εκκρεμούς (deg)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["x_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["x"], label="μη γραμμική λύση", linewidth=2.5)
plt.plot(res["time"], res["x_measured"], "*", label="μετρήσεις", markersize=0.5)
plt.legend()
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση θέσης βαγονιού (m)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the velocity response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["v_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["v"], label="μη γραμμική λύση", linewidth=2.5)
plt.legend()
plt.ylabel("v (m/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση ταχύτητας βαγονιού (m/s)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["omega_1_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["omega_1"], label="μη γραμμική λύση", linewidth=2.5)
plt.legend()
plt.ylabel("$ω_1$ (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση γωνιακής ταχύτητας εκκρεμούς $ω_1$ (rad/s)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the rotational velocity response
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["omega_2_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
plt.plot(res["time"], res["omega_2"], label="μη γραμμική λύση", linewidth=2.5)
plt.legend()
plt.ylabel("$ω_2$ (rad/s)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απόκριση γωνιακής ταχύτητας εκκρεμούς $ω_2$ (rad/s)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], res["force"], label="force controller", where="post", linewidth=2.5)
plt.ylabel("u(t) (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Απαιτούμενη δύναμη για τον έλεγχο (Ν)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()

# Plot the required force from the LQR
plt.figure(figsize=[6, 4], dpi=150)
plt.step(res["time"], error_states, where="post", linewidth=2.5)
plt.ylabel("$e^x$ (-)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Σφάλμα καταστάσεων", fontweight='bold', size=BIGGER_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.show()

# # Plot the estimated error for position of cart
# plt.figure(figsize=[6, 4], dpi=150)
# plt.plot(res["time"][10:], res["error"]["x"][10:], label="$le^{x}$ (m)", linewidth=2.5)
# plt.plot(res["time"][10:], res["error"]["theta_1"][10:], label="$le^{θ_1}$ (rad)", linewidth=2.5)
# plt.plot(res["time"][10:], res["error"]["theta_2"][10:], label="$le^{θ_2}$ (rad)", linewidth=2.5)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.legend()
# plt.ylabel("le")
# plt.xlabel("time (s)")
# plt.xlim(start_time, final_time)
# plt.title("Τοπικά σφάλματα μετρήσεων", fontweight='bold', size=BIGGER_SIZE)
# plt.grid()
# plt.show()

# Plot the estimated error for force by controller
plt.figure(figsize=[6, 4], dpi=150)
plt.plot(res["time"], res["error"]["force"], linewidth=2.5)
plt.ylabel("$le^u$ (N)")
plt.xlabel("time (s)")
plt.xlim(start_time, final_time)
plt.title("Τοπικό σφάλμα απαιτούμενης δύναμης ελεγκτή (N)", fontweight='bold', size=BIGGER_SIZE)
plt.grid()
plt.show()