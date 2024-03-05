import math
import numpy as np
import matplotlib.pyplot as plt
from csv import reader


# reading csv
file = "C:/Users/Кирилл/Desktop/НЕ ТРОГАТЬ/24_Чита.csv"


data = []
with open(file, "r") as rfile:
    for row in reader(rfile):
        data.append(row)
    data.pop(0)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])


for row in data:
    print(row)


# Lagrange polynomial
def getLagrangePolynomial(x_val, y_val):
    basePolynomial = []
    for i in range(len(x_val)):
        n = 1
        for j in range(len(x_val)):
            if j != i:
                n *= ((x-x_val[j])/(x_val[i]-x_val[j]))
        basePolynomial.append(n)

    def LagrangePolynomial():
        result = 0
        for e in range(len(y_val)):
            result += y_val[e] * basePolynomial[e]
        return result
    return LagrangePolynomial


t = 3
year = 13
xValues = []
yValues = []
count = -1

while len(xValues) != 12:
    if data[year][t] < 999.9:
        xValues.append(data[year][0])
        yValues.append(data[year][t])
        count += 1
    year += 1
x = np.arange(xValues[0], xValues[count] + 0.1, 0.1)


polynomial = getLagrangePolynomial(xValues, yValues)
plt.plot(x, polynomial())
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Lagrange polynomial")
plt.grid()
plt.show()
# First Newton interpolation formula
def getNewtonPolynomial(x_val, y_val):
    n = len(x_val) - 1
    div_diff = [[0] * (n + 1) for _ in range(n + 1)]
    div_diff[0] = y_val

    for j in range(1, n + 1):
        for i in range(n - j + 1):
            div_diff[j][i] = (div_diff[j - 1][i + 1] - div_diff[j - 1][i]) / (x_val[i + j] - x_val[i])

    def NewtonPolynomial(x):
        result = div_diff[0][0]
        prod = 1
        for i in range(1, n + 1):
            prod *= (x - x_val[i - 1])
            result += div_diff[i][0] * prod
        return result

    return NewtonPolynomial

t = 3
year = 13
xValues = []
yValues = []
count = -1

while len(xValues) != 6:
    if data[year][t] < 999.9:
        xValues.append(data[year][0])
        yValues.append(data[year][t])
        count += 1
    year += 1

polynomial = getNewtonPolynomial(xValues, yValues)
x = np.arange(xValues[0], xValues[count] + 0.1, 0.1)
plt.plot(x, polynomial(x))
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Newton polynomial (Formula 1)")
plt.grid()
plt.show()
# Second Newton interpolation formula
def getSecondNewtonPolynomial(x_val, y_val):
    n = len(x_val) - 1
    div_diff = [[0] * (n + 1) for _ in range(n + 1)]
    div_diff[0] = y_val

    for j in range(1, n + 1):
        for i in range(n - j + 1):
            div_diff[j][i] = (div_diff[j - 1][i + 1] - div_diff[j - 1][i]) / (x_val[i + j] - x_val[i])

    def SecondNewtonPolynomial(x):
        result = div_diff[0][0]
        prod = 1
        for i in range(1, n + 1):
            prod *= (x - x_val[i - 1])
            result += div_diff[i][0] * prod / math.factorial(i)
        return result

    return SecondNewtonPolynomial

t = 3
year = 19
xValues = []
yValues = []
count = -1

while len(xValues) != 6:
    if data[year][t] < 999.9:
        xValues.append(data[year][0])
        yValues.append(data[year][t])
        count += 1
    year += 1

polynomial = getSecondNewtonPolynomial(xValues, yValues)
x = np.arange(xValues[0], xValues[count] + 0.1, 0.1)
plt.plot(x, polynomial(x))
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Newton polynomial (Formula 2)")
plt.grid()
plt.show()
# Polynomial approximation
def getPolynomialApproximation(x_val, y_val, degree):
    coefficients = np.polyfit(x_val, y_val, degree)
    polynomial = np.poly1d(coefficients)

    def PolynomialApproximation(x):
        return polynomial(x)

    return PolynomialApproximation

t = 3
xValues = []
yValues = []

for row in data:
    if row[t] < 999.9:
        xValues.append(row[0])
        yValues.append(row[t])

degree = 5
polynomial = getPolynomialApproximation(xValues, yValues, degree)
x = np.arange(min(xValues), max(xValues) + 0.1, 0.1)
plt.plot(x, polynomial(x))
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Power Polynomial Approximation")
plt.grid()
plt.show()
