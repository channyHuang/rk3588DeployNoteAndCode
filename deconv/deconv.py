import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# record
U = [4.000000000004E-05, 0.00026, 0.00037, 0.00021, 0.00015, 0.0003, 0.00041, 0.00032, 0.00037, 0.00047, 0.00055, 0.00047, 0.00031, 8.00000000000245E-05, 6.00000000000045E-05, 0.00019, 0.00043, 0.00053, 0.00044, 0.00027, 4.99999999999945E-05]
# kernel
C = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, ]
# signal
m_input = [0, 0, 0, 0, 0, 0.023809523810024, 0.023809523810024, 0.071428571430071, 0.190476190480058, 0.595238095250067, 1.00000000002007, 0.595238095250067, 0.190476190480058, 0.071428571430071, 0.023809523810024, 0.023809523810024, 0, 0, 0, 0, 0, ]

# len = 21
m_fix = [0 for _ in range(10)]
m = m_fix + m_input + m_fix

U = np.matrix(U)
A = [[0 for _ in range(21)] for _ in range(21)]
for r in range(21):
    idx = r + 20
    for c in range(21):
        A[r][c] = m[idx]
        idx -= 1
A_inv = np.linalg.inv(A)
A_inv = np.matrix(A_inv)
x = A_inv @ np.transpose(U)
# print(A_inv.shape, U.shape)
x = np.transpose(x)
# print(x)

# x = [-8.44377834e-05  1.02478506e-04  3.49252932e-04 -6.21339530e-05
#    2.14220580e-05  3.53806024e-05  3.92948886e-04 -1.32201864e-04
#    2.66546960e-04  3.85536795e-05  4.05433552e-04  4.65020018e-05
#    2.99420052e-04 -2.05183194e-04  7.81052795e-05 -7.66168932e-05
#    2.80525107e-04  2.42396422e-04  1.75106970e-04  1.59229709e-04
#   -1.00302285e-04]

# draw
x = np.array(x)
x = x[0]

y = [i - 10 for i in range(len(x))]
plt.plot(y, x)
plt.show()


# C = np.array(x)
# C = C[0]
# U = np.array(U)
# U = U[0]
# # # U = c * m
# for x_ffp in range(-10, 10):
#     print('x_ffp = ', x_ffp)
#     u_cal = 0
#     for x in range(-10, 10):
#         c_x = C[x + 10] # c(x)
#         m_xffp_x = m[x_ffp - x + 20] # m(xffp - x)
#         u_cal += c_x * m_xffp_x # \sum(c(x) * m(x_ffp - x))
#     print('u_cal = ', u_cal, ', U=', U[x_ffp + 10])