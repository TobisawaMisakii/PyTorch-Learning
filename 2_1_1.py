import torch
x = torch.arange(12)
print(x.numel())
print(x.shape)
print(x)
X = x.reshape(3, 4)
print(X)
# X = X.reshape(2, 6)
X = X.reshape(2, -1)
#reshape(-1, n) or reshpe(n, -1)
#-1 : automatically division
print(X)

# y = torch.zeros(2, 3, 4)
# y = torch.ones(2, 3, 4)
# y = torch.randn(3, 4)
Y = [[2, 1, 3], [3, 2, 4]]
y = torch.tensor(Y)
print(y)
