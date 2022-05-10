from master import MasterOptions, Master
from oscilator import Oscilator

# Create the two model objects
model_1 = Oscilator(1, 100, 1, osc_method="displacement")
model_2 = Oscilator(1, 100, 1, osc_method="displacement")
models = [model_1, model_2]

# Define connections
connections = []

# Define the master object for the co-simulation
master = Master(models, step_size=1e-3, order=1, communication_method="Gauss")

# Simulate the models
start_time = 0
final_time = 12
initial_states = [0, 1, 1, 0]
res = master.simulate(initial_states, start_time, final_time)