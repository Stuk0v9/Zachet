import numpy as np
import matplotlib.pyplot as plt

# загрузка данных из .bin-файла
data = np.fromfile("var24_z2.bin", dtype = float)

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

# расчёт спектра
f_axe = np.fft.fftfreq(n, sr)
spec = np.fft.fft(new_data, n)
shift_spec = np.fft.fftshift(spec)

# графики
plt.figure()
plt.plot(f_axe, np.abs(shift_spec)) # амплитудный спектр
plt.yscale('log')  # Устанавливаем логарифмическую шкалу по вертикальной оси
plt.xscale('log')  # Устанавливаем логарифмическую шкалу по горизонтальной оси
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда A(v)')

plt.figure()
plt.plot(t_axe, data) # начальные отсчёты (синий)
plt.plot(t_axe, new_data) # умножили на окно Гаусса (оранжевый)
plt.plot(t_axe, window) # выводим график окна Гаусса (зелёный)
plt.xlabel('Время')
plt.ylabel('Величина сигнала S(t)')

harmonics_frequencies = [f_axe[np.argmax(spec)]] # определение частоты гармоник
print("Частоты гармоник:")
print(harmonics_frequencies)

noise_start = int(0.8*len(spec))  # Начало промежутка после гармоник
noise_level = np.mean(np.abs(new_data[noise_start:])) # определение уровня шума (среднее)
print("Уровень шумов:")
print(noise_level)