import numpy as np


s = b'\x00\x01\x02\x03\x00\x01\x02\x03'


s+=b'\x0a\x0b'
print(*np.frombuffer(s, np.uint16))
print(len(s))