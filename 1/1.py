
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
# Newton's forward interpolation


def getNewtonsInterpolationF(x_val, y_val):
    fdiff = []
    for i in range(1, len(x_val)):
        res = 0
        for j in range(i + 1):
            num = 1
            for k in range(i + 1):
                if k != j:
                    num *= x_val[j] - x_val[k]
            res += y_val[j] / num
        fdiff.append(res)
    return fdiff

def NewtonsInterpolation(x, x_val, y_val, fdiff):
    result = y_val[0]
    for k in range(1, len(y_val)):
        additive = 1
        for j in range(k):
            additive *= (x - x_val[j])
        result += fdiff[k - 1] * additive
    return result

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

X = np.arange(xValues[0], xValues[count] + 0.1, 0.1)
fdiff = getNewtonsInterpolationF(xValues, yValues)

plt.plot(X, [NewtonsInterpolation(x, xValues, yValues, fdiff) for x in X])
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Newton's forward interpolation")
plt.grid()
plt.show()

# Newton's back interpolation


def getNewtonsInterpolationB(x_val, y_val):
    fdiff = []
    for i in range(1, len(x_val)):
        res = 0
        for j in range(i + 1):
            num = 1
            for k in range(i + 1):
                if k != j:
                    num *= x_val[len(x_val) - 1 - j] - x_val[len(x_val) - 1 - k]
            res += y_val[len(x_val) - 1 - j] / num
        fdiff.append(res)
    return fdiff

def NewtonsPolynomial(x, x_val, y_val, fdiff):
    result = y_val[len(x_val) - 1]
    for k in range(1, len(y_val)):
        additive = 1
        for j in range(k):
            additive *= (x - x_val[len(x_val) - 1 - j])
        result += fdiff[k - 1] * additive
    return result

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

X = np.arange(xValues[0], xValues[count] + 0.1, 0.1)
fdiff = getNewtonsInterpolationB(xValues, yValues)

plt.plot(X, [NewtonsPolynomial(x, xValues, yValues, fdiff) for x in X])
plt.plot(xValues, yValues, 'ro')
plt.suptitle("Newton's back interpolation")
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
