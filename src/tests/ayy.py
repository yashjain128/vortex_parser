import numpy as np
A = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.uint8)
print(A.view(np.uint8))
A.byteswap(inplace=True)
print(A.view(np.uint8))
