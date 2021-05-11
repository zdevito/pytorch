# this is imported by test_deploy to do some checks in python
import sys

def setup(path):
    sys.path.extend(path.strip('\n').split(':'))
    sys.path.append('build/lib')

# smoke test the numpy extension loading works
def numpy_test(x):
    import numpy as np
    xs = [np.array([x, x]), np.array([x, x])]
    for i in range(10):
        xs.append(xs[-1] + xs[-2])
    return int(xs[-1][0])
