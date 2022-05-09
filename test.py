from master import MasterOptions, Master
from oscilator import Oscilator

# Create the two model objects
model_1 = Oscilator(1, 100, 1, osc_method="displacement")
model_2 = Oscilator(1, 100, 1, osc_method="displacement")
models = [model_1, model_2]

# Define the connections of the models
connections = [(model_1, "x", model_2, "x_c"), (model_1, "v", model_2, "v_c"), 
               (model_2, "x", model_1, "x_c"), (model_2, "v", model_1, "v_c")]

# Define the master object for the co-simulation
master = Master(models, connections, step_size=1e-3, order=1, communication_method="Gauss")

# Simulate the models
res = master.simulate()