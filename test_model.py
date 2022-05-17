from single_pendulum import SinglePendulum
from numpy import cos, sin
import matplotlib.pyplot as plt

# Input function to the model
def f(t):
    return [1 * cos(0.01 * t), -1 * sin(0.01 * t)]

def main():    
    model_1 = SinglePendulum(1, 0.4, 0.5, 0.1)
    mp = model_1.get("mass_pendulum")
    mc = model_1.get("mass_cart")
    l = model_1.get("length_pendulum")
    I = model_1.get("inertia_pendulum")
    print(mp, mc, l, I)

    # Simulate the model on it's own
    res = model_1.simulate([0, 0, 0.5235, 0], 20, method="BDF")

    # Plot the resuls
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(res["time"], res["x"], label="x (m)")
    plt.plot(res["time"], res["theta"], label="theta (rad)")
    plt.ylabel("states")
    plt.xlabel("time (s)")
    plt.legend()
    plt.xlim(0, res["time"][-1])
    plt.title("Single Pendulum on Cart state response")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()