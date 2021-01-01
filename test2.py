import numpy as np

a = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], order='C')

print(a)
print(a.shape)

a.resize((4, 4))

print(a)
print(a.shape)

b = np.pad(a, [(0, 5), (0, 0)], mode='constant', constant_values=-1)
print(b)
