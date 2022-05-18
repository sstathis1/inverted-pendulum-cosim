from single_pendulum import SinglePendulum
from scipy.linalg import solve_continuous_are as ricatti
import numpy as np
from numpy import cos, sin 
from math import pi
import matplotlib.pyplot as plt

def main():    
    model_1 = SinglePendulum(1.5, 0.5, 0.5, 0)
    mp = model_1.get("mass_pendulum")
    mc = model_1.get("mass_cart")
    l = model_1.get("length_pendulum")
    I = model_1.get("inertia_pendulum")
    b = model_1.get("friction_coefficient")
    print(f"mass pendulum: {mp}, mass cart: {mc}, length pendulum: {l}, inertia pendulum: {I}, friction_coefficient: {b}")

    # Get the gains for the controller using LQR continuous infinite time horizon
    K = gains(mp, mc, l, I, b)

    # Simulate the model on it's own
    res = model_1.simulate([0, 0, 20 * pi / 180, 0], 8, input=lambda x: -K.dot(x))

    # Create an animation of the results
    model_1.animate(res, savefig=True)

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

def gains(mp, mc, l, I, b):
    g = 9.81
    d0 = mp + mc
    d1 = mp * l**2 + I
    d2 = mp * l
    p = d1 * d0 - d2**2
    Q = np.diag([5000000, 5000000, 500000, 500000])
    R = 0.1
    A = np.array([[0, 1, 0, 0],
                    [0, - d1 * b / p, - d2**2 * g, 0],
                    [0, 0, 0, 1],
                    [0, d2 * b, d0 * d2 * g, 0]])
    B = np.array([0, d1 / p, 0, - d2 / p]).reshape(-1, 1)
    P = ricatti(A, B, Q, R)
    K = 1 / R * B.T.dot(P)
    return K

if __name__ == "__main__":
    main()