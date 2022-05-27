from double_pendulum import DoublePendulum
from math import pi
import matplotlib.pyplot as plt

model = DoublePendulum(1.5, 0.4, 0.4, 0.4, 0.4, 0)

res = model.simulate([0, 0, 10 * pi / 180, 0, 10 * pi / 180, 0], 10)

model.animate(res, savefig=True)

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["theta_1"] * 180 / pi)
plt.ylabel("theta_1 (deg)")
plt.xlabel("time (s)")
plt.xlim(0, res["time"][-1])
plt.title("Double Pendulum on Cart first pendulum angle response")
plt.grid()
plt.show()

# Plot the angle response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["theta_2"] * 180 / pi)
plt.ylabel("theta_2 (deg)")
plt.xlabel("time (s)")
plt.xlim(0, res["time"][-1])
plt.title("Double Pendulum on Cart second pendulum angle response")
plt.grid()
plt.show()

# Plot the position response
plt.figure(figsize=[6, 4], dpi=200)
plt.plot(res["time"], res["x"])
plt.ylabel("x (m)")
plt.xlabel("time (s)")
plt.xlim(0, res["time"][-1])
plt.title("Double Pendulum on Cart position response")
plt.grid()
plt.show()