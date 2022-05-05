from oscilator import Oscilator
from math import cos

model_1 = Oscilator(1, 100, 1)
m = model_1.get("m")
k = model_1.get("k")
c = model_1.get("c")

def f(t):
    return 10 * cos(t)

model_1.set_input(("force", f))
print(model_1.get_inputs(5))