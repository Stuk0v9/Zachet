import numpy as np
import matplotlib.pyplot as plt

#загрузка данных из .bin-файла
data = np.fromfile("var24_z1.bin", dtype = np.int16)

#вводим константы
sr = 1000
n = len(data)
t_max = n / sr
t_axe = np.linspace(0, t_max, n)
t_axe2 = np.arange(0, t_max)

#график начальных значений
plt.plot(t_axe, data, '-', linewidth = 1)
plt.grid()

#расчёт усреднения
sec = sr
data_avr = []
h = 10*sec
summ = 0
for i in range(0, h, sec):
    summ += data[0]
    avr = summ / h
    data_avr.append(avr)
    
for i in range(h, n, sec):
    summ = 0
    for j in range(i-h, i):
        summ += data[j] 
    avr = summ / h 
    data_avr.append(avr)

#график усреднения
plt.plot(t_axe2, data_avr, '-', linewidth = 1)
