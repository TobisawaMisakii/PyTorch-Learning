import torch
import torch as t
import math

#element-wise calculation
x = t.tensor([1.5, 3, -1, 0])
y = t.tensor([-2, 1, 2, 5])
print(x+y, end='\n')
print(x-y, end='\n')
print(x*y, end='\n')
print(x/y, end='\n')
print(x**y, end='\n')
print(t.exp(y), end='\n')
Y = t.zeros(y.shape)
t.exp(y, out = Y)
print(Y)

print(t.exp(y) == Y)

#concat
x = t.arange(12, dtype=float).reshape(3,4)
y = t.randn(12, dtype=float).reshape(3,4)
print(x, y, end="\n")
print(t.cat((x,y), dim=0), end='\n')
print(t.cat((x,y), dim=1), end='\n')

print(x.sum())

print("Test Git Push")
