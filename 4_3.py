import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# загрузка данных из .bin-файла
data = np.fromfile("var24_z3.bin", dtype = float)

# константы
sr = 1000
n = len(data)
t_max = n / sr
t_axe = np.linspace(0, t_max, n)

# окно Гаусса
def gauss_window(n, sigma=0.3):
    A = (n-1)/2
    x = np.arange(0, n)
    window = np.exp(-((x-A)/(sigma*A))**2 / 2)
    return window

# Создание окна Гаусса
window = gauss_window(n)

# Умножение последовательности на окно Гаусса
new_data = data * window

f_axe, t_axe, spec = spectrogram(new_data, fs=sr, window=('gaussian', 100), nperseg=1000, noverlap=999)

plt.figure() # работам с продвинутой спектрограммой
plt.pcolormesh(t_axe, f_axe, 10*np.log10(spec))  
plt.ylabel('Частота, Гц')
plt.xlabel('Время, с')
plt.colorbar(label='Уровень мощности, dB')

#plt.figure() # работаем с привычным A(v)
#plt.plot(f_axe, spec)
#plt.show()