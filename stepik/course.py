from fractions import Fraction
from sys import stdin

import numpy as np

from stepik.gaussian_elimination import matrix_solve

stream = None
try:
    stream = open('input.txt', 'r')
except:
    stream = stdin

n, m = map(int, stream.readline().split(" "))

if n >= m:
    A = []
    for i in range(n):
        A.append([item for item in map(Fraction, stream.readline().strip().split(" "))])

    A = np.array(A, Fraction)

    matrix_solve(A)
else:
    print("INF")
