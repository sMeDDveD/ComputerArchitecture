import numpy as np
import matplotlib.pyplot as b

with open('data2.txt', 'r') as data_file:
    b.xkcd()
    x = list(map(int, data_file.readline().split()))
    y = list(map(float, data_file.readline().split()))
    b.figure()
    b.plot(x, y, 'r--', label = 'Matrices dimension - 2000')
    b.xticks(x)
    b.xlabel('Number of processes')
    b.ylabel('Time(s)')
    b.title('Matrix multiplication')
    b.legend(loc='upper left')
    b.show()