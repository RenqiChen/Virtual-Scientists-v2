import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

random_values = np.random.normal(loc=0.005, scale=0.005, size=arr.shape)
random_values = np.abs(random_values)
random_values[random_values > 0.01] = 0.01
print(random_values)