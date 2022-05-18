from single_pendulum import SinglePendulum
from numpy import cos, sin 
from math import pi
import matplotlib.pyplot as plt

# Optional: Input function to the model
def f(x):
    K = []
    return - K.dot(x)

def main():    
    model_1 = SinglePendulum(1, 0.4, 0.4)
    mp = model_1.get("mass_pendulum")
    mc = model_1.get("mass_cart")
    l = model_1.get("length_pendulum")
    I = model_1.get("inertia_pendulum")
    print(f"mass pendulum: {mp}, mass cart: {mc}, length pendulum: {l}, inertia pendulum: {I}")

    # Simulate the model on it's own
    res = model_1.simulate([0, 0, 10 * pi / 180, 0], 5)

    # Create an animation of the results
    model_1.animate(res, savefig=False)

    # Plot the angle response
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(res["time"], res["theta"] * 180 / pi)
    plt.ylabel("theta (deg)")
    plt.xlabel("time (s)")
    plt.xlim(0, res["time"][-1])
    plt.title("Single Pendulum on Cart angle response")
    plt.grid()
    plt.show()

    # Plot the position response
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(res["time"], res["x"])
    plt.ylabel("x (m)")
    plt.xlabel("time (s)")
    plt.xlim(0, res["time"][-1])
    plt.title("Single Pendulum on Cart position response")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()