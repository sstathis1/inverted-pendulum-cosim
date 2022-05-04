from master import MasterOptions, Master

master = Master(step_size = 0.001, order = 2, rtol = 1e-2, communication_method = "Gauss")
options = master.options

for key in options:
    print(key, " : ", options[key])