import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

random_values = np.random.normal(loc=0.05, scale=0.05, size=arr.shape)
random_values = np.abs(random_values)
random_values[random_values > 0.1] = 0.1
print(random_values)