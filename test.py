from math import pi
import numpy as np
from master import MasterOptions, Master
from single_pendulum import SinglePendulum as Pendulum
from single_pendulum_controller import SinglePendulumController as Controller

# Create the two model objects
model_1 = Pendulum(1, 0.4, 0.4, 0.05)
model_2 = Controller(1, 0.4, 0.4, 0.05)
models = [model_1, model_2]

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=0, communication_method="Jacobi", error_controlled=False)

# Simulate the models
start_time = 0
final_time = 12
initial_states = [0, 0, 40 * pi / 180, 0, 0, 0, 40 * pi / 180, 0]
res = master.simulate(initial_states, start_time, final_time)

model_1.animate([res["x"], res["theta"]], savefig=False)