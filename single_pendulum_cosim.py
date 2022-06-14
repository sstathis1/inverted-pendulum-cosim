import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from master import Master
from single_pendulum import SinglePendulum as Pendulum
from single_pendulum_controller import SinglePendulumController as Controller

def main():
    # Covariance Matrices
    P = 1e-2* np.eye(4)
    Q = np.diag([0, 0, 0, 0, 0, 0])
    R = np.diag([1e-4, 1e-4])

    res = {}
    states_nl = {}
    states_l = {}
    error_states = {}
    for estimation_method in ["kalman"]:
        # Create the two model objects
        model_1 = Pendulum(1.5, 0.5, 0.6, 0.05)
        model_2 = Controller(1.5, 0.5, 0.6, 0.05, estimation_method=estimation_method, P=P, R=R)
        models = [model_1, model_2]

        # Define the master object for the co-simulation
        master = Master(models, step_size=1e-2, order=0, communication_method="Gauss", 
                        error_controlled=True, is_parallel=False)

        # Simulate the models
        start_time = 0
        final_time = 3
        initial_states = [0, 0, 40 * pi / 180, 0, 0, 0, 40 * pi / 180, 0]

        # Start the timer
        start_timer = time.perf_counter()

        res[estimation_method] = master.simulate(initial_states, start_time, final_time)

        end_timer = time.perf_counter()
        print(f"Co-Simulation finished correctly in : {end_timer-start_timer} second(s)")

        # Save the state error
        states_nl[estimation_method] = np.array([res[estimation_method]["x"], res[estimation_method]["v"], 
                                                 res[estimation_method]["theta"], res[estimation_method]["omega"]])
        states_l[estimation_method] = np.array([res[estimation_method]["x_linear"], res[estimation_method]["v_linear"], 
                                                res[estimation_method]["theta_linear"], res[estimation_method]["omega_linear"]])
        error_states[estimation_method] = np.linalg.norm(states_nl[estimation_method] - states_l[estimation_method], axis=0)

    # model_1.animate(res, savefig=False)
    
    plot_results(start_time, res, error_states)


def plot_results(start_time, res, error_states):
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
    plt.step(res["kalman"]["time"], res["kalman"]["theta_linear"] * 180 / pi, label="γραμμική λύση", where="post", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["theta"] * 180 / pi, label="μη γραμμική λύση", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["theta_measured"] * 180 / pi, "*", label="μετρήσεις", markersize=1.5)
    plt.legend()
    plt.ylabel("$θ$ (deg)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Απόκριση γωνίας θ εκκρεμούς (deg)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the position response
    plt.figure(figsize=[6, 4], dpi=150)
    plt.step(res["kalman"]["time"], res["kalman"]["x_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["x"], label="μη γραμμική λύση", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["x_measured"], "*", label="μετρήσεις", markersize=1.5)
    plt.legend()
    plt.ylabel("x (m)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Απόκριση θέσης βαγονιού (m)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the velocity response
    plt.figure(figsize=[6, 4], dpi=150)
    plt.step(res["kalman"]["time"], res["kalman"]["v_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["v"], label="μη γραμμική λύση", linewidth=2.5)
    plt.legend()
    plt.ylabel("v (m/s)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Απόκριση ταχύτητας βαγονιού (m/s)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the rotational velocity response
    plt.figure(figsize=[6, 4], dpi=150)
    plt.step(res["kalman"]["time"], res["kalman"]["omega_linear"], label="γραμμική λύση", where="post", linewidth=2.5)
    plt.plot(res["kalman"]["time"], res["kalman"]["omega"], label="μη γραμμική λύση", linewidth=2.5)
    plt.legend()
    plt.ylabel("$ω$ (rad/s)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Απόκριση γωνιακής ταχύτητας εκκρεμούς (rad/s)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the required force from the LQR
    plt.figure(figsize=[6, 4], dpi=150)
    plt.step(res["kalman"]["time"], res["kalman"]["force"], label="force controller predictive", where="post", linewidth=2.5)
    plt.ylabel("u(t) (N)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Απαιτούμενη δύναμη για τον έλεγχο (Ν)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the required force from the LQR
    plt.figure(figsize=[6, 4], dpi=150)
    plt.step(res["kalman"]["time"], error_states["kalman"], label="προβλεπτικός εκτιμητής", where="post", linewidth=2.5)
    # plt.step(res["current"]["time"], error_states["current"], label="σύγχρονος εκτιμητής", where="post", linewidth=2.5)
    # plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel("$e^x$")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Σφάλμα καταστάσεων", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the estimated error for position of cart
    plt.figure(figsize=[6, 4], dpi=150)
    plt.plot(res["kalman"]["time"], res["kalman"]["error"]["x"], label="προβλεπτικός εκτιμητής", linewidth=2.5)
    # plt.plot(res["current"]["time"], res["current"]["error"]["x"], label="σύγχρονος εκτιμητής", linewidth=2.5)
    # plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel("$le^x$ (m)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Τοπικό σφάλμα θέσης βαγονιού (m)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the estimated error for angle of pendulum
    plt.figure(figsize=[6, 4], dpi=150)
    plt.plot(res["kalman"]["time"], res["kalman"]["error"]["theta"], label="προβλεπτικός εκτιμητής", linewidth=2.5)
    # plt.plot(res["current"]["time"], res["current"]["error"]["theta"], label="σύγχρονος εκτιμητής", linewidth=2.5)
    plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel("$le^{θ}$ (rad)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Τοπικό σφάλμα γωνίας εκκρεμούς (rad)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

    # Plot the estimated error for force by controller
    plt.figure(figsize=[6, 4], dpi=150)
    plt.plot(res["kalman"]["time"], res["kalman"]["error"]["force"], label="προβλεπτικός εκτιμητής", linewidth=2.5)
    # plt.plot(res["current"]["time"], res["current"]["error"]["force"], label="σύγχρονος εκτιμητής", linewidth=2.5)
    # plt.legend()
    plt.ylabel("$le^u$ (N)")
    plt.xlabel("time (s)")
    plt.xlim(start_time, res["kalman"]["time"][-1])
    plt.title("Τοπικό σφάλμα δύναμης ελεγκτή (N)", fontweight='bold', size=BIGGER_SIZE)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()