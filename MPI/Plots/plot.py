import numpy as a
import matplotlib.pyplot as b

with open('data1.txt', 'r') as data_file:
    x = list(map(int, data_file.readline().split()))
    y = [list(map(float, data_file.readline().split())) for i in range(4)]
    b.figure()
    b.plot(x, y[0], 'r--', label = 'n = 1')
    b.plot(x, y[1], 'b', label = 'n = 2')
    b.plot(x, y[2], 'g', label = 'n = 4')
    b.plot(x, y[3], 'black', label = 'n = 8')
    b.xlabel('Размерность матриц')
    b.ylabel('Время')
    b.title('Умножение матриц')
    b.legend(loc='upper left')
    b.show()