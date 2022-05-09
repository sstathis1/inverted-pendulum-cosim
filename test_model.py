from oscilator import Oscilator
from math import cos, sin
import matplotlib.pyplot as plt

# Input function to the model
def f(t):
    return [1 * cos(0.01 * t), -1 * sin(0.01 * t)]

def main():    
    model_1 = Oscilator(1, 100, 1)
    m = model_1.get("m")
    k = model_1.get("k")
    c = model_1.get("c")

    print(f"The states of the model before the simulation are: {model_1.states}")
    print(f"The outputs of the model before the simulation are: {model_1.output}")
    print(f"The time variable for the model before the simulation is set to be: {model_1.time}")
    print()
    # Simulate the model on it's own
    res = model_1.simulate([0, 1], f, 10)

    print(f"The states of the model after the simulation are: {model_1.states}")
    print(f"The outputs of the model after the simulation are: {model_1.output}")
    print(f"The time variable for the model after the simulation is set to be: {model_1.time}")
    print(f"The input to the model at the end of the simulation is: {model_1.input}")

    # Plot the resuls
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(res["time"], res["x"], label="x")
    plt.plot(res["time"], res["v"], label="v")
    plt.ylabel("x (m), v (m/s)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.xlim(0, res["time"][-1])
    plt.title("One-Degree of Freedom Linear Oscilator response")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()