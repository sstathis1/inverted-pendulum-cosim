from double_pendulum_controller import DoublePendulumController
from double_pendulum import DoublePendulum
from math import pi
import matplotlib.pyplot as plt

model_non_linear = DoublePendulum(1.5, 0.5, 0.75, 0.5, 0.75, 0)
model_linear = DoublePendulumController(1.5, 0.5, 0.75, 0.5, 0.75, 0)

res_nl = model_non_linear.simulate([0, 0, 15 * pi / 180, 0, 15 * pi / 180, 0], 5)
res_l = model_linear.simulate([0, 0, 15 * pi / 180, 0, 15 * pi / 180, 0], 5)

model_non_linear.animate(res_nl, savefig=False)

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res_l["time"], res_l["theta_1"] * 180 / pi, label="linear")
plt.plot(res_nl["time"], res_nl["theta_1"] * 180 / pi, label="non-linear")
plt.ylabel("theta_1 (deg)")
plt.xlabel("time (s)")
plt.xlim(0, res_l["time"][-1])
plt.title("Double Pendulum on Cart first pendulum angle response")
plt.grid()
plt.show()

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res_l["time"], res_l["theta_2"] * 180 / pi, label="linear")
plt.plot(res_nl["time"], res_nl["theta_2"] * 180 / pi, label="non-linear")
plt.ylabel("theta_2 (deg)")
plt.xlabel("time (s)")
plt.xlim(0, res_l["time"][-1])
plt.title("Double Pendulum on Cart second pendulum angle response")
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res_l["time"], res_l["x"] * 180 / pi, label="linear")
plt.plot(res_nl["time"], res_nl["x"] * 180 / pi, label="non-linear")
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(0, res_l["time"][-1])
plt.title("Double Pendulum on Cart position response")
plt.grid()
plt.show()