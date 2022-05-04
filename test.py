from master import MasterOptions

test = MasterOptions(step_size = 0.001, order = 2, rtol = 1e-2, communication_method = "Gauss")
options = test.get_options()

for key in options:
    print(key, " : ", options[key])