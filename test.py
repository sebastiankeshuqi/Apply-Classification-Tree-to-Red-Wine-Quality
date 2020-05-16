from math import exp,factorial
res = 0
for i in range(7):
    res += factorial(100)/factorial(i)/factorial(100-i)*(0.05**i)*(0.95**(100-i))
print(1-res)